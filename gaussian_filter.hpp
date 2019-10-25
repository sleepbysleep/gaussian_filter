/*
  Copyright (C) 2014 Hoyoung Lee

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
#include <limits>
#include <iostream>
#include <cstring>
#include <float.h>

#include <cpixmap.hpp>
#include <cchunk.hpp>

#if !defined(USE_SIMD)

template <typename T>
void blurGaussian3x3Kernel(cpixmap<T>& dst, cpixmap<T>& src)
{
  assert(std::numeric_limits<T>::is_integer);
  //assert(!std::numeric_limits<T>::is_signed);
  assert(std::numeric_limits<T>::digits < std::numeric_limits<int>::digits);
  //assert(dst.isMatched((dimension)src));
  
  assert(dst.getWidth() == src.getWidth());
  assert(dst.getHeight() == src.getHeight());
  assert(dst.getBands() >= src.getBands());

  for (size_t z = 0; z < src.getBands(); ++z) {
    window3x3_frame<T> win3x3(src);
    win3x3.draftFrame(src, z);
    
    for (size_t y = 0; y < src.getHeight(); ++y) {
      T *dst_line = dst.getLine(y, z);
#pragma omp parallel for
      for (size_t x = 0; x < src.getWidth(); ++x) {
	dst_line[x] = static_cast<T>(
	  ((float)win3x3(y-1, x-1)*1)/16 + ((float)win3x3(y-1, x)*2)/16 + ((float)win3x3(y-1, x+1)*1)/16 +
	  ((float)win3x3(y,   x-1)*2)/16 + ((float)win3x3(y,   x)*4)/16 + ((float)win3x3(y,   x+1)*2)/16 +
	  ((float)win3x3(y+1, x-1)*1)/16 + ((float)win3x3(y+1, x)*2)/16 + ((float)win3x3(y+1, x+1)*1)/16);
	  
	  /*
	  (win3x3(y-1, x-1)>>4) + (win3x3(y-1, x)>>3) + (win3x3(y-1, x+1)>>4) +
	  (win3x3(y,   x-1)>>3) + (win3x3(y,   x)>>2) + (win3x3(y,   x+1)>>3) +
	  (win3x3(y+1, x-1)>>4) + (win3x3(y+1, x)>>3) + (win3x3(y+1, x+1)>>4);
	  */
      }
      win3x3.shiftFrame(src, z);
    }
  }
}

#else
# include "gaussian_filter.SIMD.hpp"
#endif

typedef enum {
  UNDIRECTIONAL = 0,
  HORIZONTAL = 1,
  VERTICAL = 2, 
  DIAGONAL1 = 3,
  DIAGONAL2 = 4,
  NR_DIRECTION = 5
} direction_t;

template <typename T>
void blurDirectionalGaussian3x1Kernel(cpixmap<T>& dst, cpixmap<uint8_t>& dirmap, cpixmap<T>& src)
{
  assert(std::numeric_limits<T>::is_integer);
  //assert(!std::numeric_limits<T>::is_signed);
  assert(std::numeric_limits<T>::digits < std::numeric_limits<int>::digits);
  assert(dst.isMatched(src));
  assert(dst.isMatched(dirmap));

  for (size_t z = 0; z < src.getBands(); ++z) {
    window3x3_frame<T> win3x3(src);
    win3x3.draftFrame(src, z);
    
    for (size_t y = 0; y < src.getHeight(); ++y) {
      uint8_t *dirLine = dirmap.getLine(y, z);
#pragma omp parallel for
      for (size_t x = 0; x < src.getWidth(); ++x) {
	int hDiff = std::abs((int)win3x3(y, x-1) - (int)win3x3(y, x+1));
	int hDir = hDiff + (hDiff>>2) + (hDiff>>3) + (hDiff>>5);
	
	int vDiff = std::abs((int)win3x3(y-1, x) - (int)win3x3(y+1, x));
	int vDir = vDiff + (vDiff>>2) + (vDiff>>3) + (vDiff>>5);
	
	//int hDir = hDiff + (hDiff/4) + (hDiff/8) + (hDiff/32);
	//int vDir = vDiff + (vDiff/4) + (vDiff/8) + (vDiff/32);
	int d1Diff = std::abs((int)win3x3(y-1, x+1) - (int)win3x3(y+1, x-1));
	int d2Diff = std::abs((int)win3x3(y-1, x-1) - (int)win3x3(y+1, x+1));

	if (hDir > vDir) {
	  if (hDir > d1Diff) {
	    if (hDir > d2Diff) dirLine[x] = HORIZONTAL; // H
	    else dirLine[x] = DIAGONAL2; // D2
	  } else {
	    if (d1Diff > d2Diff) dirLine[x] = DIAGONAL1; // D1
	    else dirLine[x] = DIAGONAL2; // D2
	  }
	} else {
	  if (vDir > d1Diff) {
	    if (vDir > d2Diff) dirLine[x] = VERTICAL; // V
	    else dirLine[x] = DIAGONAL2; // D2
	  } else {
	    if (d1Diff > d2Diff) dirLine[x] = DIAGONAL1; // D1
	    else dirLine[x] = DIAGONAL2; // D2
	  }
	}
      }
      win3x3.shiftFrame(src, z);
    }
  }

  for (size_t z = 0; z < src.getBands(); ++z) {
    window3x3_frame<uint8_t> dir3x3(dirmap);
    dir3x3.draftFrame(dirmap, z);

    window3x3_frame<T> img3x3(src);
    img3x3.draftFrame(src, z);
    
    for (size_t y = 0; y < src.getHeight(); ++y) {
      T *dstLine = dst.getLine(y, z);
#pragma omp parallel for
      for (size_t x = 0; x < src.getWidth(); ++x) {
	int dirCount[NR_DIRECTION] = {0,0,0,0,0};
	//std::memset(dirCount, 0, NR_DIRECTION * sizeof(int));
	dirCount[dir3x3(y-1, x-1)]++, dirCount[dir3x3(y-1, x)]++, dirCount[dir3x3(y-1, x+1)]++;
	dirCount[dir3x3(y, x-1)]++, dirCount[dir3x3(y, x)]++, dirCount[dir3x3(y, x+1)]++;
	dirCount[dir3x3(y+1, x-1)]++, dirCount[dir3x3(y+1, x)]++, dirCount[dir3x3(y+1, x+1)]++;

	/*
	int maxarg = 1;
	for (int i = HORIZONTAL+1; i < NR_DIRECTION; i++) {
	  if (dirCount[i] > dirCount[maxarg]) maxarg = i;
	}
	*/
	uint8_t maxarg;
	if (dirCount[HORIZONTAL] > dirCount[VERTICAL]) {
	  if (dirCount[HORIZONTAL] > dirCount[DIAGONAL1]) {
	    if (dirCount[HORIZONTAL] > dirCount[DIAGONAL2]) maxarg = HORIZONTAL;
	    else maxarg = DIAGONAL2;
	  } else {
	    if (dirCount[DIAGONAL1] > dirCount[DIAGONAL2]) maxarg = DIAGONAL1;
	    else maxarg = DIAGONAL2;
	  }
	} else {
	  if (dirCount[VERTICAL] > dirCount[DIAGONAL1]) {
	    if (dirCount[VERTICAL] > dirCount[DIAGONAL2]) maxarg = VERTICAL;
	    else maxarg = DIAGONAL2;
	  } else {
	    if (dirCount[DIAGONAL1] > dirCount[DIAGONAL2]) maxarg = DIAGONAL1;
	    else maxarg = DIAGONAL2;
	  }
	}
	//maxarg = maxarg;
	switch (maxarg) {
	case HORIZONTAL:
	  dstLine[x] = (img3x3(y, x-1)>>2) + (img3x3(y, x)>>1) + (img3x3(y, x+1)>>2); break;
	case VERTICAL:
	  dstLine[x] = (img3x3(y-1, x)>>2) + (img3x3(y, x)>>1) + (img3x3(y+1, x)>>2); break;
	case DIAGONAL1:
	  dstLine[x] = (img3x3(y-1, x+1)>>2) + (img3x3(y, x)>>1) + (img3x3(y+1, x-1)>>2); break;
	case DIAGONAL2:
	  dstLine[x] = (img3x3(y-1, x-1)>>2) + (img3x3(y, x)>>1) + (img3x3(y+1, x+1)>>2); break;
	default: abort(); break;
	}
      }

      dir3x3.shiftFrame(dirmap, z);
      img3x3.shiftFrame(src, z);
    }
  }
}
