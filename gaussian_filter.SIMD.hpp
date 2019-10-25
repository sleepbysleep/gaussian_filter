/*
  Copyright (C) 2017 Hoyoung Lee

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once

#include <cassert>
#include <iostream>
#include <cstring>
#include <cstdint>

#include <cpixmap.hpp>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
# define MAX_VECTOR_SIZE 512
# include <vectorclass/vectorclass.h>
# if INSTRSET < 2
#  error "Unsupported x86-SIMD! Please comment USE_SIMD on!"
# endif
#elif defined(__GNUC__) && defined (__ARM_NEON__)
# include <arm_neon.h>
#else
# error "Undefined SIMD!"
#endif

inline void blurGaussian3x3Kernel(cpixmap<uint8_t>& dst, cpixmap<uint8_t>& src)
{
  assert(dst.getWidth() == src.getWidth());
  assert(dst.getHeight() == src.getHeight());
  assert(dst.getBands() >= src.getBands());

  for (size_t z = 0; z < src.getBands(); ++z) {
    window3x3_frame<uint8_t> win3x3(src);
    win3x3.draftFrame(src, z);
    
    for (size_t y = 0; y < src.getHeight(); y++) {
      uint8_t *dstLine = dst.getLine(y, z);
      uint8_t *prevLine = win3x3.getPrevLine();
      uint8_t *currLine = win3x3.getCurrLine();
      uint8_t *nextLine = win3x3.getNextLine();

      /*
      nwVec|nnVec|neVec
      -----+-----+-----
      wwVec|ooVec|eeVec
      -----+-----+-----
      swVec|ssVec|seVec
      */
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 8 // AVXx - 256bits
      for (size_t x = 0; x < src.getWidth(); x += 32) {
	Vec32uc nwVec, nnVec, neVec;
	nwVec.load(&prevLine[(int)x-1]), nnVec.load(&prevLine[(int)x+0]), neVec.load(&prevLine[(int)x+1]);
	Vec32uc wwVec, ooVec, eeVec;
	wwVec.load(&currLine[(int)x-1]), ooVec.load(&currLine[(int)x+0]), eeVec.load(&currLine[(int)x+0]);
	Vec32uc swVec, ssVec, seVec;
	swVec.load(&nextLine[(int)x-1]), ssVec.load(&nextLine[(int)x+0]), seVec.load(&nextLine[(int)x+1]);
	Vec32uc dstVec =
	  (nwVec>>4) + (nnVec>>3) + (neVec>>4) +
	  (wwVec>>3) + (ooVec>>2) + (eeVec>>3) +
	  (swVec>>4) + (ssVec>>3) + (seVec>>4);
	dstVec.store(&dstLine[x]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t x = 0; x < src.getWidth(); x += 16) {
	Vec16uc nwVec, nnVec, neVec;
	nwVec.load(&prevLine[(int)x-1]), nnVec.load(&prevLine[(int)x+0]), neVec.load(&prevLine[(int)x+1]);
	Vec16uc wwVec, ooVec, eeVec;
	wwVec.load(&currLine[(int)x-1]), ooVec.load(&currLine[(int)x+0]), eeVec.load(&currLine[(int)x+0]);
	Vec16uc swVec, ssVec, seVec;
	swVec.load(&nextLine[(int)x-1]), ssVec.load(&nextLine[(int)x+0]), seVec.load(&nextLine[(int)x+1]);
	Vec16uc dstVec =
	  (nwVec>>4) + (nnVec>>3) + (neVec>>4) +
	  (wwVec>>3) + (ooVec>>2) + (eeVec>>3) +
	  (swVec>>4) + (ssVec>>3) + (seVec>>4);
	dstVec.store(&dstLine[x]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t x = 0; x < src.getWidth(); x += 16) {
	uint8x16_t nwVec, nnVec, neVec;
	nwVec = vld1q_u8((const uint8_t *)&prevLine[(int)x-1]);
	nnVec = vld1q_u8((const uint8_t *)&prevLine[(int)x+0]);
	neVec = vld1q_u8((const uint8_t *)&prevLine[(int)x+1]);
	uint8x16_t wwVec, ooVec, eeVec;
	wwVec = vld1q_u8((const uint8_t *)&currLine[(int)x-1]);
	ooVec = vld1q_u8((const uint8_t *)&currLine[(int)x+0]);
	eeVec = vld1q_u8((const uint8_t *)&currLine[(int)x+1]);
	uint8x16_t swVec, ssVec, seVec;
	swVec = vld1q_u8((const uint8_t *)&nextLine[(int)x-1]);
	ssVec = vld1q_u8((const uint8_t *)&nextLine[(int)x+0]);
	seVec = vld1q_u8((const uint8_t *)&nextLine[(int)x+1]);
	
	uint8x16_t sumVec;
	sumVec = vdupq_n_u8((uint8_t)0);
	sumVec = vsra_n_u8(sumVec, nwVec, 4);
	sumVec = vsra_n_u8(sumVec, nnVec, 3);
	sumVec = vsra_n_u8(sumVec, neVec, 4);
	sumVec = vsra_n_u8(sumVec, wwVec, 3);
	sumVec = vsra_n_u8(sumVec, ooVec, 2);
	sumVec = vsra_n_u8(sumVec, eeVec, 3);
	sumVec = vsra_n_u8(sumVec, swVec, 4);
	sumVec = vsra_n_u8(sumVec, ssVec, 3);
	sumVec = vsra_n_u8(sumVec, seVec, 4);
	vst1q_u8((uint8_t *)&dstLine[x], sumVec);
      }
#endif
      win3x3.shiftFrame(src, z);
    }
  }
}

inline void blurGaussian3x3Kernel(cpixmap<int8_t>& dst, cpixmap<int8_t>& src)
{
  assert(dst.getWidth() == src.getWidth());
  assert(dst.getHeight() == src.getHeight());
  assert(dst.getBands() >= src.getBands());

  for (size_t z = 0; z < src.getBands(); ++z) {
    window3x3_frame<int8_t> win3x3(src);
    win3x3.draftFrame(src, z);
    
    for (size_t y = 0; y < src.getHeight(); y++) {
      int8_t *dstLine = dst.getLine(y, z);
      int8_t *prevLine = win3x3.getPrevLine();
      int8_t *currLine = win3x3.getCurrLine();
      int8_t *nextLine = win3x3.getNextLine();

      /*
      nwVec|nnVec|neVec
      -----+-----+-----
      wwVec|ooVec|eeVec
      -----+-----+-----
      swVec|ssVec|seVec
      */
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 8 // AVXx - 256bits
      for (size_t x = 0; x < src.getWidth(); x += 32) {
	Vec32c nwVec, nnVec, neVec;
	nwVec.load(&prevLine[(int)x-1]), nnVec.load(&prevLine[(int)x+0]), neVec.load(&prevLine[(int)x+1]);
	Vec32c wwVec, ooVec, eeVec;
	wwVec.load(&currLine[(int)x-1]), ooVec.load(&currLine[(int)x+0]), eeVec.load(&currLine[(int)x+0]);
	Vec32c swVec, ssVec, seVec;
	swVec.load(&nextLine[(int)x-1]), ssVec.load(&nextLine[(int)x+0]), seVec.load(&nextLine[(int)x+1]);
	Vec32c dstVec =
	  (nwVec>>4) + (nnVec>>3) + (neVec>>4) +
	  (wwVec>>3) + (ooVec>>2) + (eeVec>>3) +
	  (swVec>>4) + (ssVec>>3) + (seVec>>4);
	dstVec.store(&dstLine[x]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t x = 0; x < src.getWidth(); x += 16) {
	Vec16c nwVec, nnVec, neVec;
	nwVec.load(&prevLine[(int)x-1]), nnVec.load(&prevLine[(int)x+0]), neVec.load(&prevLine[(int)x+1]);
	Vec16c wwVec, ooVec, eeVec;
	wwVec.load(&currLine[(int)x-1]), ooVec.load(&currLine[(int)x+0]), eeVec.load(&currLine[(int)x+0]);
	Vec16c swVec, ssVec, seVec;
	swVec.load(&nextLine[(int)x-1]), ssVec.load(&nextLine[(int)x+0]), seVec.load(&nextLine[(int)x+1]);
	Vec16c dstVec =
	  (nwVec>>4) + (nnVec>>3) + (neVec>>4) +
	  (wwVec>>3) + (ooVec>>2) + (eeVec>>3) +
	  (swVec>>4) + (ssVec>>3) + (seVec>>4);
	dstVec.store(&dstLine[x]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t x = 0; x < src.getWidth(); x += 16) {
	int8x16_t nwVec, nnVec, neVec;
	nwVec = vld1q_s8((const int8_t *)&prevLine[(int)x-1]);
	nnVec = vld1q_s8((const int8_t *)&prevLine[(int)x+0]);
	neVec = vld1q_s8((const int8_t *)&prevLine[(int)x+1]);
	int8x16_t wwVec, ooVec, eeVec;
	wwVec = vld1q_s8((const int8_t *)&currLine[(int)x-1]);
	ooVec = vld1q_s8((const int8_t *)&currLine[(int)x+0]);
	eeVec = vld1q_s8((const int8_t *)&currLine[(int)x+1]);
	int8x16_t swVec, ssVec, seVec;
	swVec = vld1q_s8((const int8_t *)&nextLine[(int)x-1]);
	ssVec = vld1q_s8((const int8_t *)&nextLine[(int)x+0]);
	seVec = vld1q_s8((const int8_t *)&nextLine[(int)x+1]);
	
	int8x16_t sumVec;
	sumVec = vdupq_n_s8((int8_t)0);
	sumVec = vsra_n_s8(sumVec, nwVec, 4);
	sumVec = vsra_n_s8(sumVec, nnVec, 3);
	sumVec = vsra_n_s8(sumVec, neVec, 4);
	sumVec = vsra_n_s8(sumVec, wwVec, 3);
	sumVec = vsra_n_s8(sumVec, ooVec, 2);
	sumVec = vsra_n_s8(sumVec, eeVec, 3);
	sumVec = vsra_n_s8(sumVec, swVec, 4);
	sumVec = vsra_n_s8(sumVec, ssVec, 3);
	sumVec = vsra_n_s8(sumVec, seVec, 4);
	vst1q_s8((int8_t *)&dstLine[x], sumVec);
      }
#endif
      win3x3.shiftFrame(src, z);
    }
  }
}

inline void blurGaussian3x3Kernel(cpixmap<uint16_t>& dst, cpixmap<uint16_t>& src)
{
  assert(dst.getWidth() == src.getWidth());
  assert(dst.getHeight() == src.getHeight());
  assert(dst.getBands() >= src.getBands());

  for (size_t z = 0; z < src.getBands(); ++z) {
    window3x3_frame<uint16_t> win3x3(src);
    win3x3.draftFrame(src, z);
    
    for (size_t y = 0; y < src.getHeight(); y++) {
      uint16_t *dstLine = dst.getLine(y, z);
      uint16_t *prevLine = win3x3.getPrevLine();
      uint16_t *currLine = win3x3.getCurrLine();
      uint16_t *nextLine = win3x3.getNextLine();

      /*
      nwVec|nnVec|neVec
      -----+-----+-----
      wwVec|ooVec|eeVec
      -----+-----+-----
      swVec|ssVec|seVec
      */
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 8 // AVXx - 256bits
      for (size_t x = 0; x < src.getWidth(); x += 16) {
	Vec16us nwVec, nnVec, neVec;
	nwVec.load(&prevLine[(int)x-1]), nnVec.load(&prevLine[(int)x+0]), neVec.load(&prevLine[(int)x+1]);
	Vec16us wwVec, ooVec, eeVec;
	wwVec.load(&currLine[(int)x-1]), ooVec.load(&currLine[(int)x+0]), eeVec.load(&currLine[(int)x+0]);
	Vec16us swVec, ssVec, seVec;
	swVec.load(&nextLine[(int)x-1]), ssVec.load(&nextLine[(int)x+0]), seVec.load(&nextLine[(int)x+1]);
	Vec16us dstVec =
	  (nwVec>>4) + (nnVec>>3) + (neVec>>4) +
	  (wwVec>>3) + (ooVec>>2) + (eeVec>>3) +
	  (swVec>>4) + (ssVec>>3) + (seVec>>4);
	dstVec.store(&dstLine[x]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t x = 0; x < src.getWidth(); x += 8) {
	Vec8us nwVec, nnVec, neVec;
	nwVec.load(&prevLine[(int)x-1]), nnVec.load(&prevLine[(int)x+0]), neVec.load(&prevLine[(int)x+1]);
	Vec8us wwVec, ooVec, eeVec;
	wwVec.load(&currLine[(int)x-1]), ooVec.load(&currLine[(int)x+0]), eeVec.load(&currLine[(int)x+0]);
	Vec8us swVec, ssVec, seVec;
	swVec.load(&nextLine[(int)x-1]), ssVec.load(&nextLine[(int)x+0]), seVec.load(&nextLine[(int)x+1]);
	Vec8us dstVec =
	  (nwVec>>4) + (nnVec>>3) + (neVec>>4) +
	  (wwVec>>3) + (ooVec>>2) + (eeVec>>3) +
	  (swVec>>4) + (ssVec>>3) + (seVec>>4);
	dstVec.store(&dstLine[x]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t x = 0; x < src.getWidth(); x += 8) {
	uint16x8_t nwVec, nnVec, neVec;
	nwVec = vld1q_u16((const uint16_t *)&prevLine[(int)x-1]);
	nnVec = vld1q_u16((const uint16_t *)&prevLine[(int)x+0]);
	neVec = vld1q_u16((const uint16_t *)&prevLine[(int)x+1]);
	uint16x8_t wwVec, ooVec, eeVec;
	wwVec = vld1q_u16((const uint16_t *)&currLine[(int)x-1]);
	ooVec = vld1q_u16((const uint16_t *)&currLine[(int)x+0]);
	eeVec = vld1q_u16((const uint16_t *)&currLine[(int)x+1]);
	uint16x8_t swVec, ssVec, seVec;
	swVec = vld1q_u16((const uint16_t *)&nextLine[(int)x-1]);
	ssVec = vld1q_u16((const uint16_t *)&nextLine[(int)x+0]);
	seVec = vld1q_u16((const uint16_t *)&nextLine[(int)x+1]);
	
	uint16x8_t sumVec;
	sumVec = vdupq_n_u16((uint16_t)0);
	sumVec = vsra_n_u16(sumVec, nwVec, 4);
	sumVec = vsra_n_u16(sumVec, nnVec, 3);
	sumVec = vsra_n_u16(sumVec, neVec, 4);
	sumVec = vsra_n_u16(sumVec, wwVec, 3);
	sumVec = vsra_n_u16(sumVec, ooVec, 2);
	sumVec = vsra_n_u16(sumVec, eeVec, 3);
	sumVec = vsra_n_u16(sumVec, swVec, 4);
	sumVec = vsra_n_u16(sumVec, ssVec, 3);
	sumVec = vsra_n_u16(sumVec, seVec, 4);
	vst1q_u16((uint16_t *)&dstLine[x], sumVec);
      }
#endif
      win3x3.shiftFrame(src, z);
    }
  }
}

inline void blurGaussian3x3Kernel(cpixmap<int16_t>& dst, cpixmap<int16_t>& src)
{
  assert(dst.getWidth() == src.getWidth());
  assert(dst.getHeight() == src.getHeight());
  assert(dst.getBands() >= src.getBands());

  for (size_t z = 0; z < src.getBands(); ++z) {
    window3x3_frame<int16_t> win3x3(src);
    win3x3.draftFrame(src, z);
    
    for (size_t y = 0; y < src.getHeight(); y++) {
      int16_t *dstLine = dst.getLine(y, z);
      int16_t *prevLine = win3x3.getPrevLine();
      int16_t *currLine = win3x3.getCurrLine();
      int16_t *nextLine = win3x3.getNextLine();

      /*
      nwVec|nnVec|neVec
      -----+-----+-----
      wwVec|ooVec|eeVec
      -----+-----+-----
      swVec|ssVec|seVec
      */
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 8 // AVXx - 256bits
      for (size_t x = 0; x < src.getWidth(); x += 16) {
	Vec16s nwVec, nnVec, neVec;
	nwVec.load(&prevLine[(int)x-1]), nnVec.load(&prevLine[(int)x+0]), neVec.load(&prevLine[(int)x+1]);
	Vec16s wwVec, ooVec, eeVec;
	wwVec.load(&currLine[(int)x-1]), ooVec.load(&currLine[(int)x+0]), eeVec.load(&currLine[(int)x+0]);
	Vec16s swVec, ssVec, seVec;
	swVec.load(&nextLine[(int)x-1]), ssVec.load(&nextLine[(int)x+0]), seVec.load(&nextLine[(int)x+1]);
	Vec16s dstVec =
	  (nwVec>>4) + (nnVec>>3) + (neVec>>4) +
	  (wwVec>>3) + (ooVec>>2) + (eeVec>>3) +
	  (swVec>>4) + (ssVec>>3) + (seVec>>4);
	dstVec.store(&dstLine[x]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t x = 0; x < src.getWidth(); x += 8) {
	Vec8s nwVec, nnVec, neVec;
	nwVec.load(&prevLine[(int)x-1]), nnVec.load(&prevLine[(int)x+0]), neVec.load(&prevLine[(int)x+1]);
	Vec8s wwVec, ooVec, eeVec;
	wwVec.load(&currLine[(int)x-1]), ooVec.load(&currLine[(int)x+0]), eeVec.load(&currLine[(int)x+0]);
	Vec8s swVec, ssVec, seVec;
	swVec.load(&nextLine[(int)x-1]), ssVec.load(&nextLine[(int)x+0]), seVec.load(&nextLine[(int)x+1]);
	Vec8s dstVec =
	  (nwVec>>4) + (nnVec>>3) + (neVec>>4) +
	  (wwVec>>3) + (ooVec>>2) + (eeVec>>3) +
	  (swVec>>4) + (ssVec>>3) + (seVec>>4);
	dstVec.store(&dstLine[x]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t x = 0; x < src.getWidth(); x += 8) {
	int16x8_t nwVec, nnVec, neVec;
	nwVec = vld1q_s16((const int16_t *)&prevLine[(int)x-1]);
	nnVec = vld1q_s16((const int16_t *)&prevLine[(int)x+0]);
	neVec = vld1q_s16((const int16_t *)&prevLine[(int)x+1]);
	int16x8_t wwVec, ooVec, eeVec;
	wwVec = vld1q_s16((const int16_t *)&currLine[(int)x-1]);
	ooVec = vld1q_s16((const int16_t *)&currLine[(int)x+0]);
	eeVec = vld1q_s16((const int16_t *)&currLine[(int)x+1]);
	int16x8_t swVec, ssVec, seVec;
	swVec = vld1q_s16((const int16_t *)&nextLine[(int)x-1]);
	ssVec = vld1q_s16((const int16_t *)&nextLine[(int)x+0]);
	seVec = vld1q_s16((const int16_t *)&nextLine[(int)x+1]);
	
	int16x8_t sumVec;
	sumVec = vdupq_n_s16((int16_t)0);
	sumVec = vsra_n_s16(sumVec, nwVec, 4);
	sumVec = vsra_n_s16(sumVec, nnVec, 3);
	sumVec = vsra_n_s16(sumVec, neVec, 4);
	sumVec = vsra_n_s16(sumVec, wwVec, 3);
	sumVec = vsra_n_s16(sumVec, ooVec, 2);
	sumVec = vsra_n_s16(sumVec, eeVec, 3);
	sumVec = vsra_n_s16(sumVec, swVec, 4);
	sumVec = vsra_n_s16(sumVec, ssVec, 3);
	sumVec = vsra_n_s16(sumVec, seVec, 4);
	vst1q_s16((int16_t *)&dstLine[x], sumVec);
      }
#endif
      win3x3.shiftFrame(src, z);
    }
  }
}

inline void blurGaussian3x3Kernel(cpixmap<uint32_t>& dst, cpixmap<uint32_t>& src)
{
  assert(dst.getWidth() == src.getWidth());
  assert(dst.getHeight() == src.getHeight());
  assert(dst.getBands() >= src.getBands());

  for (size_t z = 0; z < src.getBands(); ++z) {
    window3x3_frame<uint32_t> win3x3(src);
    win3x3.draftFrame(src, z);
    
    for (size_t y = 0; y < src.getHeight(); y++) {
      uint32_t *dstLine = dst.getLine(y, z);
      uint32_t *prevLine = win3x3.getPrevLine();
      uint32_t *currLine = win3x3.getCurrLine();
      uint32_t *nextLine = win3x3.getNextLine();

      /*
      nwVec|nnVec|neVec
      -----+-----+-----
      wwVec|ooVec|eeVec
      -----+-----+-----
      swVec|ssVec|seVec
      */
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 9 // AVX512 - 512bits
      for (size_t x = 0; x < src.getWidth(); x += 16) {
	Vec16ui nwVec, nnVec, neVec;
	nwVec.load(&prevLine[(int)x-1]), nnVec.load(&prevLine[(int)x+0]), neVec.load(&prevLine[(int)x+1]);
	Vec16ui wwVec, ooVec, eeVec;
	wwVec.load(&currLine[(int)x-1]), ooVec.load(&currLine[(int)x+0]), eeVec.load(&currLine[(int)x+0]);
	Vec16ui swVec, ssVec, seVec;
	swVec.load(&nextLine[(int)x-1]), ssVec.load(&nextLine[(int)x+0]), seVec.load(&nextLine[(int)x+1]);
	Vec16ui dstVec =
	  (nwVec>>4) + (nnVec>>3) + (neVec>>4) +
	  (wwVec>>3) + (ooVec>>2) + (eeVec>>3) +
	  (swVec>>4) + (ssVec>>3) + (seVec>>4);
	dstVec.store(&dstLine[x]);
      }
# elif INSTRSET >= 8 // AVX2 - 256bits
      for (size_t x = 0; x < src.getWidth(); x += 8) {
	Vec8ui nwVec, nnVec, neVec;
	nwVec.load(&prevLine[(int)x-1]), nnVec.load(&prevLine[(int)x+0]), neVec.load(&prevLine[(int)x+1]);
	Vec8ui wwVec, ooVec, eeVec;
	wwVec.load(&currLine[(int)x-1]), ooVec.load(&currLine[(int)x+0]), eeVec.load(&currLine[(int)x+0]);
	Vec8ui swVec, ssVec, seVec;
	swVec.load(&nextLine[(int)x-1]), ssVec.load(&nextLine[(int)x+0]), seVec.load(&nextLine[(int)x+1]);
	Vec8ui dstVec =
	  (nwVec>>4) + (nnVec>>3) + (neVec>>4) +
	  (wwVec>>3) + (ooVec>>2) + (eeVec>>3) +
	  (swVec>>4) + (ssVec>>3) + (seVec>>4);
	dstVec.store(&dstLine[x]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t x = 0; x < src.getWidth(); x += 4) {
	Vec4ui nwVec, nnVec, neVec;
	nwVec.load(&prevLine[(int)x-1]), nnVec.load(&prevLine[(int)x+0]), neVec.load(&prevLine[(int)x+1]);
	Vec4ui wwVec, ooVec, eeVec;
	wwVec.load(&currLine[(int)x-1]), ooVec.load(&currLine[(int)x+0]), eeVec.load(&currLine[(int)x+0]);
	Vec4ui swVec, ssVec, seVec;
	swVec.load(&nextLine[(int)x-1]), ssVec.load(&nextLine[(int)x+0]), seVec.load(&nextLine[(int)x+1]);
	Vec4ui dstVec =
	  (nwVec>>4) + (nnVec>>3) + (neVec>>4) +
	  (wwVec>>3) + (ooVec>>2) + (eeVec>>3) +
	  (swVec>>4) + (ssVec>>3) + (seVec>>4);
	dstVec.store(&dstLine[x]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t x = 0; x < src.getWidth(); x += 4) {
	uint32x4_t nwVec, nnVec, neVec;
	nwVec = vld1q_u32((const uint32_t *)&prevLine[(int)x-1]);
	nnVec = vld1q_u32((const uint32_t *)&prevLine[(int)x+0]);
	neVec = vld1q_u32((const uint32_t *)&prevLine[(int)x+1]);
	uint32x4_t wwVec, ooVec, eeVec;
	wwVec = vld1q_u32((const uint32_t *)&currLine[(int)x-1]);
	ooVec = vld1q_u32((const uint32_t *)&currLine[(int)x+0]);
	eeVec = vld1q_u32((const uint32_t *)&currLine[(int)x+1]);
	uint32x4_t swVec, ssVec, seVec;
	swVec = vld1q_u32((const uint32_t *)&nextLine[(int)x-1]);
	ssVec = vld1q_u32((const uint32_t *)&nextLine[(int)x+0]);
	seVec = vld1q_u32((const uint32_t *)&nextLine[(int)x+1]);
	
	uint32x4_t sumVec;
	sumVec = vdupq_n_u32((uint32_t)0);
	sumVec = vsra_n_u32(sumVec, nwVec, 4);
	sumVec = vsra_n_u32(sumVec, nnVec, 3);
	sumVec = vsra_n_u32(sumVec, neVec, 4);
	sumVec = vsra_n_u32(sumVec, wwVec, 3);
	sumVec = vsra_n_u32(sumVec, ooVec, 2);
	sumVec = vsra_n_u32(sumVec, eeVec, 3);
	sumVec = vsra_n_u32(sumVec, swVec, 4);
	sumVec = vsra_n_u32(sumVec, ssVec, 3);
	sumVec = vsra_n_u32(sumVec, seVec, 4);
	vst1q_u32((uint32_t *)&dstLine[x], sumVec);
      }
#endif
      win3x3.shiftFrame(src, z);
    }
  }
}

inline void blurGaussian3x3Kernel(cpixmap<int32_t>& dst, cpixmap<int32_t>& src)
{
  assert(dst.getWidth() == src.getWidth());
  assert(dst.getHeight() == src.getHeight());
  assert(dst.getBands() >= src.getBands());

  for (size_t z = 0; z < src.getBands(); ++z) {
    window3x3_frame<int32_t> win3x3(src);
    win3x3.draftFrame(src, z);
    
    for (size_t y = 0; y < src.getHeight(); y++) {
      int32_t *dstLine = dst.getLine(y, z);
      int32_t *prevLine = win3x3.getPrevLine();
      int32_t *currLine = win3x3.getCurrLine();
      int32_t *nextLine = win3x3.getNextLine();

      /*
      nwVec|nnVec|neVec
      -----+-----+-----
      wwVec|ooVec|eeVec
      -----+-----+-----
      swVec|ssVec|seVec
      */
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 9 // AVX512 - 512bits
      for (size_t x = 0; x < src.getWidth(); x += 16) {
	Vec16i nwVec, nnVec, neVec;
	nwVec.load(&prevLine[(int)x-1]), nnVec.load(&prevLine[(int)x+0]), neVec.load(&prevLine[(int)x+1]);
	Vec16i wwVec, ooVec, eeVec;
	wwVec.load(&currLine[(int)x-1]), ooVec.load(&currLine[(int)x+0]), eeVec.load(&currLine[(int)x+0]);
	Vec16i swVec, ssVec, seVec;
	swVec.load(&nextLine[(int)x-1]), ssVec.load(&nextLine[(int)x+0]), seVec.load(&nextLine[(int)x+1]);
	Vec16i dstVec =
	  (nwVec>>4) + (nnVec>>3) + (neVec>>4) +
	  (wwVec>>3) + (ooVec>>2) + (eeVec>>3) +
	  (swVec>>4) + (ssVec>>3) + (seVec>>4);
	dstVec.store(&dstLine[x]);
      }
# elif INSTRSET >= 8 // AVX2 - 256bits
      for (size_t x = 0; x < src.getWidth(); x += 8) {
	Vec8i nwVec, nnVec, neVec;
	nwVec.load(&prevLine[(int)x-1]), nnVec.load(&prevLine[(int)x+0]), neVec.load(&prevLine[(int)x+1]);
	Vec8i wwVec, ooVec, eeVec;
	wwVec.load(&currLine[(int)x-1]), ooVec.load(&currLine[(int)x+0]), eeVec.load(&currLine[(int)x+0]);
	Vec8i swVec, ssVec, seVec;
	swVec.load(&nextLine[(int)x-1]), ssVec.load(&nextLine[(int)x+0]), seVec.load(&nextLine[(int)x+1]);
	Vec8i dstVec =
	  (nwVec>>4) + (nnVec>>3) + (neVec>>4) +
	  (wwVec>>3) + (ooVec>>2) + (eeVec>>3) +
	  (swVec>>4) + (ssVec>>3) + (seVec>>4);
	dstVec.store(&dstLine[x]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t x = 0; x < src.getWidth(); x += 4) {
	Vec4i nwVec, nnVec, neVec;
	nwVec.load(&prevLine[(int)x-1]), nnVec.load(&prevLine[(int)x+0]), neVec.load(&prevLine[(int)x+1]);
	Vec4i wwVec, ooVec, eeVec;
	wwVec.load(&currLine[(int)x-1]), ooVec.load(&currLine[(int)x+0]), eeVec.load(&currLine[(int)x+0]);
	Vec4i swVec, ssVec, seVec;
	swVec.load(&nextLine[(int)x-1]), ssVec.load(&nextLine[(int)x+0]), seVec.load(&nextLine[(int)x+1]);
	Vec4i dstVec =
	  (nwVec>>4) + (nnVec>>3) + (neVec>>4) +
	  (wwVec>>3) + (ooVec>>2) + (eeVec>>3) +
	  (swVec>>4) + (ssVec>>3) + (seVec>>4);
	dstVec.store(&dstLine[x]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t x = 0; x < src.getWidth(); x += 4) {
	int32x4_t nwVec, nnVec, neVec;
	nwVec = vld1q_s32((const int32_t *)&prevLine[(int)x-1]);
	nnVec = vld1q_s32((const int32_t *)&prevLine[(int)x+0]);
	neVec = vld1q_s32((const int32_t *)&prevLine[(int)x+1]);
	int32x4_t wwVec, ooVec, eeVec;
	wwVec = vld1q_s32((const int32_t *)&currLine[(int)x-1]);
	ooVec = vld1q_s32((const int32_t *)&currLine[(int)x+0]);
	eeVec = vld1q_s32((const int32_t *)&currLine[(int)x+1]);
	int32x4_t swVec, ssVec, seVec;
	swVec = vld1q_s32((const int32_t *)&nextLine[(int)x-1]);
	ssVec = vld1q_s32((const int32_t *)&nextLine[(int)x+0]);
	seVec = vld1q_s32((const int32_t *)&nextLine[(int)x+1]);
	
	int32x4_t sumVec;
	sumVec = vdupq_n_s32((int32_t)0);
	sumVec = vsra_n_s32(sumVec, nwVec, 4);
	sumVec = vsra_n_s32(sumVec, nnVec, 3);
	sumVec = vsra_n_s32(sumVec, neVec, 4);
	sumVec = vsra_n_s32(sumVec, wwVec, 3);
	sumVec = vsra_n_s32(sumVec, ooVec, 2);
	sumVec = vsra_n_s32(sumVec, eeVec, 3);
	sumVec = vsra_n_s32(sumVec, swVec, 4);
	sumVec = vsra_n_s32(sumVec, ssVec, 3);
	sumVec = vsra_n_s32(sumVec, seVec, 4);
	vst1q_s32((int32_t *)&dstLine[x], sumVec);
      }
#endif
      win3x3.shiftFrame(src, z);
    }
  }
}


