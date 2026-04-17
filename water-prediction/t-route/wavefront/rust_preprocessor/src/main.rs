//! Parallel CSV preprocessor for t-route forcings.
//!
//! Pivots 142K+ `nex-<id>_output.csv` files into a single `(timesteps x reaches)`
//! binary tensor compatible with our GPU kernel's forcings.bin format.
//!
//! Uses rayon for data-parallel reads, memmap2 for zero-copy file access.
//! No GIL, no per-file schema overhead, straight ASCII parsing.
//!
//! Format of each input CSV:
//!     `<ts>, <YYYY-MM-DD HH:MM:SS>, <qlat_f32>`
//!
//! Output forcings.bin:
//!     int32 n_timesteps
//!     float32 qlat[n_timesteps, n_reaches]
//!     float32 qup0[n_reaches]    (zeros)
//!     float32 qdp0[n_reaches]    (zeros)
//!     float32 dp0[n_reaches]     (half-meter default)

use byteorder::{LittleEndian, WriteBytesExt};
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

fn strip_wbid_from_filename(name: &str) -> Option<i64> {
    // "nex-1328591_output.csv" -> 1328591
    let rest = name.strip_prefix("nex-")?;
    let end = rest.find('_').or_else(|| rest.find('.'))?;
    rest[..end].parse().ok()
}

/// Parse one nex CSV from a byte slice into a pre-allocated [f32; timesteps] buffer.
/// Lines look like: "0, 2017-01-01 00:00:00, 0.00582754\n"
fn parse_csv_bytes(data: &[u8], timesteps: usize, out: &mut [f32]) -> usize {
    let mut matched = 0;
    let mut i = 0;
    while i < data.len() {
        let mut j = i;
        while j < data.len() && data[j] != b'\n' {
            j += 1;
        }
        let line = &data[i..j];
        i = j + 1;
        if line.len() < 5 {
            continue;
        }
        // Skip leading spaces
        let mut k = 0;
        while k < line.len() && line[k] == b' ' {
            k += 1;
        }
        let ts_start = k;
        while k < line.len() && line[k].is_ascii_digit() {
            k += 1;
        }
        if k == ts_start {
            continue;
        }
        let ts: usize = match std::str::from_utf8(&line[ts_start..k])
            .ok()
            .and_then(|s| s.parse().ok())
        {
            Some(v) => v,
            None => continue,
        };
        // Value is after the last comma
        let last_comma = match line.iter().rposition(|&b| b == b',') {
            Some(p) => p,
            None => continue,
        };
        let mut v_start = last_comma + 1;
        while v_start < line.len() && line[v_start] == b' ' {
            v_start += 1;
        }
        let val_str = match std::str::from_utf8(&line[v_start..]) {
            Ok(s) => s.trim(),
            Err(_) => continue,
        };
        let val: f32 = match val_str.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if ts < timesteps {
            out[ts] = val;
            matched += 1;
        }
    }
    matched
}

/// Load a numpy .npy file containing an int64 array.
fn load_i64_npy(path: &Path) -> std::io::Result<Vec<i64>> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 10];
    file.read_exact(&mut header)?;
    let header_len = u16::from_le_bytes([header[8], header[9]]) as usize;
    let mut header_data = vec![0u8; header_len];
    file.read_exact(&mut header_data)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    assert!(data.len() % 8 == 0);
    let n = data.len() / 8;
    let mut out = Vec::with_capacity(n);
    for chunk in data.chunks_exact(8) {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(chunk);
        out.push(i64::from_le_bytes(arr));
    }
    Ok(out)
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <csv_dir> <topo_to_wbid.npy> <out_forcings.bin> [n_timesteps=24] [threads=auto]",
            args[0]
        );
        std::process::exit(1);
    }
    let csv_dir = &args[1];
    let wbid_path = Path::new(&args[2]);
    let out_path = &args[3];
    let n_timesteps: usize = if args.len() >= 5 {
        args[4].parse().unwrap_or(24)
    } else {
        24
    };
    let n_threads: usize = if args.len() >= 6 {
        args[5].parse().unwrap_or(0)
    } else {
        0
    };
    if n_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build_global()
            .ok();
    }
    let effective_threads = rayon::current_num_threads();

    let t0 = Instant::now();

    let topo_to_wbid = load_i64_npy(wbid_path)?;
    let n_reaches = topo_to_wbid.len();
    let mut wbid_to_topo: HashMap<i64, u32> = HashMap::with_capacity(n_reaches * 2);
    for (i, &w) in topo_to_wbid.iter().enumerate() {
        wbid_to_topo.insert(w, i as u32);
    }
    let t1 = Instant::now();
    println!(
        "[rust] loaded topo_to_wbid ({} reaches) in {:.1?}",
        n_reaches,
        t1 - t0
    );

    let mut paths: Vec<PathBuf> = Vec::with_capacity(150_000);
    for entry in fs::read_dir(csv_dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = match path.file_name().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        if name.starts_with("nex-") && name.ends_with("_output.csv") {
            paths.push(path);
        }
    }
    let t2 = Instant::now();
    println!("[rust] listed {} files in {:.1?}", paths.len(), t2 - t1);

    let matched = AtomicUsize::new(0);
    let missing = AtomicUsize::new(0);

    let parsed: Vec<Option<(u32, Vec<f32>)>> = paths
        .par_iter()
        .map(|path| {
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => {
                    missing.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            };
            let wbid = match strip_wbid_from_filename(&name) {
                Some(v) => v,
                None => {
                    missing.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            };
            let tidx = match wbid_to_topo.get(&wbid) {
                Some(&i) => i,
                None => {
                    missing.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            };
            let file = match File::open(path) {
                Ok(f) => f,
                Err(_) => {
                    missing.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            };
            let mmap = match unsafe { Mmap::map(&file) } {
                Ok(m) => m,
                Err(_) => {
                    missing.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            };
            let mut vals = vec![0.0f32; n_timesteps];
            parse_csv_bytes(&mmap[..], n_timesteps, &mut vals);
            matched.fetch_add(1, Ordering::Relaxed);
            Some((tidx, vals))
        })
        .collect();

    let t3 = Instant::now();
    println!(
        "[rust] parsed {} csv (missing {}) in {:.1?} using {} threads ({:.0} files/s)",
        matched.load(Ordering::Relaxed),
        missing.load(Ordering::Relaxed),
        t3 - t2,
        effective_threads,
        paths.len() as f64 / (t3 - t2).as_secs_f64()
    );

    let mut qlat = vec![0.0f32; n_timesteps * n_reaches];
    for entry in parsed.into_iter().flatten() {
        let (tidx, vals) = entry;
        for ts in 0..n_timesteps {
            qlat[ts * n_reaches + tidx as usize] = vals[ts];
        }
    }
    let t4 = Instant::now();
    println!("[rust] scattered into output tensor in {:.1?}", t4 - t3);

    let out_file = File::create(out_path)?;
    let mut bw = BufWriter::new(out_file);
    bw.write_i32::<LittleEndian>(n_timesteps as i32)?;
    for &v in &qlat {
        bw.write_f32::<LittleEndian>(v)?;
    }
    for _ in 0..n_reaches {
        bw.write_f32::<LittleEndian>(0.0)?;
    }
    for _ in 0..n_reaches {
        bw.write_f32::<LittleEndian>(0.0)?;
    }
    for _ in 0..n_reaches {
        bw.write_f32::<LittleEndian>(0.5)?;
    }
    bw.flush()?;
    drop(bw);

    let t5 = Instant::now();
    let filesize = fs::metadata(out_path)?.len();
    println!(
        "[rust] wrote {} ({:.1} MB) in {:.1?}",
        out_path,
        filesize as f64 / 1e6,
        t5 - t4
    );
    println!("[rust] TOTAL end-to-end: {:.1?}", t5 - t0);

    Ok(())
}
