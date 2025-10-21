/*
 * ============================================================================
 *  Project: SpiralReality / Tiger Optimizer
 *  Copyright (c) 2025 Ryo ∴ SpiralArchitect and SpiralReality
 *
 *  This file is part of SpiralReality.
 *
 *  SpiralReality is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Affero General Public License as published
 *  by the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  SpiralReality is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the GNU Affero General Public License for more details.
 *
 *  You should have received a copy of the GNU Affero General Public License
 *  along with SpiralReality.  If not, see <https://www.gnu.org/licenses/>.
 * ============================================================================
 */

//! Tiger Optimizer acceleration routines implemented in Rust.
#![allow(clippy::missing_safety_doc)]

use std::slice;

#[inline(always)]
fn softsign_scalar(val: f32, tau: f32) -> f32 {
    let denom = val.abs() + tau;
    if denom == 0.0 {
        0.0
    } else {
        val / denom
    }
}

#[inline(always)]
fn softsign_scalar_f64(val: f64, tau: f64) -> f64 {
    let denom = val.abs() + tau;
    if denom == 0.0 {
        0.0
    } else {
        val / denom
    }
}

#[no_mangle]
pub unsafe extern "C" fn tiger_softsign_out_f32(
    dst: *mut f32,
    src: *const f32,
    len: usize,
    tau: f32,
) {
    if dst.is_null() || src.is_null() || len == 0 {
        return;
    }
    let dst = slice::from_raw_parts_mut(dst, len);
    let src = slice::from_raw_parts(src, len);
    for i in 0..len {
        dst[i] = softsign_scalar(src[i], tau);
    }
}

#[no_mangle]
pub unsafe extern "C" fn tiger_softsign_out_f64(
    dst: *mut f64,
    src: *const f64,
    len: usize,
    tau: f64,
) {
    if dst.is_null() || src.is_null() || len == 0 {
        return;
    }
    let dst = slice::from_raw_parts_mut(dst, len);
    let src = slice::from_raw_parts(src, len);
    for i in 0..len {
        dst[i] = softsign_scalar_f64(src[i], tau);
    }
}

#[no_mangle]
pub unsafe extern "C" fn tiger_rms_f32(src: *const f32, len: usize) -> f32 {
    if src.is_null() || len == 0 {
        return 0.0;
    }
    let src = slice::from_raw_parts(src, len);
    let mut acc = 0.0f64;
    for &v in src.iter() {
        let fv = v as f64;
        acc += fv * fv;
    }
    ((acc / len as f64).sqrt()) as f32
}

#[no_mangle]
pub unsafe extern "C" fn tiger_rms_f64(src: *const f64, len: usize) -> f64 {
    if src.is_null() || len == 0 {
        return 0.0;
    }
    let src = slice::from_raw_parts(src, len);
    let mut acc = 0.0f64;
    for &v in src.iter() {
        acc += v * v;
    }
    (acc / len as f64).sqrt()
}

#[no_mangle]
pub unsafe extern "C" fn tiger_norm_f32(src: *const f32, len: usize) -> f32 {
    if src.is_null() || len == 0 {
        return 0.0;
    }
    let src = slice::from_raw_parts(src, len);
    let mut acc = 0.0f64;
    for &v in src.iter() {
        let fv = v as f64;
        acc += fv * fv;
    }
    acc.sqrt() as f32
}

#[no_mangle]
pub unsafe extern "C" fn tiger_norm_f64(src: *const f64, len: usize) -> f64 {
    if src.is_null() || len == 0 {
        return 0.0;
    }
    let src = slice::from_raw_parts(src, len);
    let mut acc = 0.0f64;
    for &v in src.iter() {
        acc += v * v;
    }
    acc.sqrt()
}
