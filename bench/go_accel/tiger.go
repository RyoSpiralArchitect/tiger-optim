package main

/*
#include <stddef.h>
*/
import "C"

import (
    "math"
    "unsafe"
)

func float32Slice(ptr unsafe.Pointer, length C.size_t) []float32 {
    if ptr == nil || length == 0 {
        return []float32{}
    }
    return unsafe.Slice((*float32)(ptr), int(length))
}

func float64Slice(ptr unsafe.Pointer, length C.size_t) []float64 {
    if ptr == nil || length == 0 {
        return []float64{}
    }
    return unsafe.Slice((*float64)(ptr), int(length))
}

func softsign32(dst, src []float32, tau float32) {
    if len(dst) != len(src) {
        return
    }
    for i, x := range src {
        denom := float32(math.Abs(float64(x))) + tau
        dst[i] = x / denom
    }
}

func softsign64(dst, src []float64, tau float64) {
    if len(dst) != len(src) {
        return
    }
    for i, x := range src {
        denom := math.Abs(x) + tau
        dst[i] = x / denom
    }
}

func rms32(src []float32) float32 {
    if len(src) == 0 {
        return 0
    }
    var sum float64
    for _, x := range src {
        sum += float64(x) * float64(x)
    }
    mean := sum / float64(len(src))
    return float32(math.Sqrt(mean))
}

func rms64(src []float64) float64 {
    if len(src) == 0 {
        return 0
    }
    var sum float64
    for _, x := range src {
        sum += x * x
    }
    mean := sum / float64(len(src))
    return math.Sqrt(mean)
}

func norm32(src []float32) float32 {
    if len(src) == 0 {
        return 0
    }
    var sum float64
    for _, x := range src {
        sum += float64(x) * float64(x)
    }
    return float32(math.Sqrt(sum))
}

func norm64(src []float64) float64 {
    if len(src) == 0 {
        return 0
    }
    var sum float64
    for _, x := range src {
        sum += x * x
    }
    return math.Sqrt(sum)
}

//export TigerSoftsignOutF32
func TigerSoftsignOutF32(dstPtr unsafe.Pointer, srcPtr unsafe.Pointer, length C.size_t, tau C.float) {
    dst := float32Slice(dstPtr, length)
    src := float32Slice(srcPtr, length)
    softsign32(dst, src, float32(tau))
}

//export TigerSoftsignOutF64
func TigerSoftsignOutF64(dstPtr unsafe.Pointer, srcPtr unsafe.Pointer, length C.size_t, tau C.double) {
    dst := float64Slice(dstPtr, length)
    src := float64Slice(srcPtr, length)
    softsign64(dst, src, float64(tau))
}

//export TigerRmsF32
func TigerRmsF32(srcPtr unsafe.Pointer, length C.size_t) C.float {
    src := float32Slice(srcPtr, length)
    return C.float(rms32(src))
}

//export TigerRmsF64
func TigerRmsF64(srcPtr unsafe.Pointer, length C.size_t) C.double {
    src := float64Slice(srcPtr, length)
    return C.double(rms64(src))
}

//export TigerNormF32
func TigerNormF32(srcPtr unsafe.Pointer, length C.size_t) C.float {
    src := float32Slice(srcPtr, length)
    return C.float(norm32(src))
}

//export TigerNormF64
func TigerNormF64(srcPtr unsafe.Pointer, length C.size_t) C.double {
    src := float64Slice(srcPtr, length)
    return C.double(norm64(src))
}

func main() {}
