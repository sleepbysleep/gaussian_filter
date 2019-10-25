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

#include <iostream>
#include <sstream>
#include <cstring>
//#include <cmemory>
#include <cassert>
#include <cstdint>

#include "cregion.hpp"

#define QWORD_ALIGN(bytes) (((bytes) + 7) & -8)

template <typename T>
class cpixmap : public cregion<size_t> {
  //
public:
  cpixmap(void);
  cpixmap(size_t w, size_t h, size_t b = 1);
  cpixmap(const cpixmap& pixmap);
  cpixmap(const cregion& dim);
  virtual ~cpixmap(void);
  T *getImage(size_t z = 0) const;
  T *getLine(size_t y, size_t z = 0) const;
  T& getPixel(size_t x, size_t y, size_t z = 0) const;
  void putPixel(T val, size_t x, size_t y, size_t z = 0);
  void setResolution(size_t w, size_t h, size_t b = 1);
  bool isMatched(const cpixmap& pixmap) const;
  bool isMatched(const cregion& a) const;
  bool isMatched(size_t w, size_t h, size_t b = 1) const;
  void readVLine(T *line, size_t len, size_t x, size_t y, size_t z = 0) const;
  void readHLine(T *line, size_t len, size_t x, size_t y, size_t z = 0) const;
  void flipHorizontally(void);
  void flipVertically(void);
  void lshiftPixel(size_t bits = 1);
  void rshiftPixel(size_t bits = 1);
  //  cpixmap<T> operator=(const cpixmap<T>& m);
  T& operator() (size_t z, size_t y, size_t x) { return *(T *)(m_buffer + z*m_band_stride + y*m_height_stride + x*sizeof(T)); }
  T& operator() (size_t y, size_t x) { return *(T *)(m_buffer + y*m_height_stride + x*sizeof(T)); }

  enum RGB_COLOR {
    BLUE_BAND = 0,
    GREEN_BAND = 1,
    RED_BAND = 2,
    RGB_BANDS = 3
  };
  
private:
  //void reallocate(size_t w, size_t h);
  void reallocate(size_t w, size_t h, size_t b = 0);
  size_t m_height_stride;
  size_t m_band_stride;
  uint8_t *m_buffer;
};

template <typename T> 
cpixmap<T>::cpixmap(void)
  : m_height_stride(0), m_band_stride(0), m_buffer(NULL) {}

template <typename T>
cpixmap<T>::cpixmap(size_t w, size_t h, size_t b)
  : cregion(w, h, b), m_height_stride(0), m_band_stride(0), m_buffer(NULL)
{
  //setResolution(w, h, b);
  reallocate(w, h, b);
}

template <typename T>
cpixmap<T>::cpixmap(const cpixmap& pixmap)
  : m_height_stride(0), m_band_stride(0), m_buffer(NULL)
{
  const cregion dim = static_cast<const cregion>(pixmap);
  setResolution(dim.getWidth(), dim.getHeight(), dim.getBands());
}
  
template <typename T>
cpixmap<T>::cpixmap(const cregion& dim)
  : m_height_stride(0), m_band_stride(0), m_buffer(NULL)
{
  setResolution(dim.getWidth(), dim.getHeight(), dim.getBands());
}

template <typename T>
cpixmap<T>::~cpixmap(void)
{
  //std::cout << static_cast<void *>(this) << " paraent" <<std::endl;
  //std::cout << static_cast<void *>(m_buffer) << " is freed!" << std::endl;
  if (m_buffer) delete [] reinterpret_cast<double *>(m_buffer);
  m_buffer = NULL;
}

template <typename T>
void cpixmap<T>::setResolution(size_t w, size_t h, size_t b)
{
  cregion::setResolution(w, h, b);
  reallocate(w, h, b);
}

template <typename T>
void cpixmap<T>::reallocate(size_t w, size_t h, size_t b)
{
  size_t bytes;

  m_height_stride = QWORD_ALIGN(w * sizeof(T)); //+ 7) & -8; // 8 bytes alignment
  m_band_stride = h * m_height_stride;
  
  bytes = b * m_band_stride;

  if (m_buffer) delete (double *)m_buffer;
  m_buffer = reinterpret_cast<uint8_t *>(new double[(bytes + 7) / 8]);
  assert(m_buffer);
  memset(m_buffer, 0, bytes);
  //std::cout << static_cast<void *>(this) << " paraent" <<std::endl;
  //std::cout << bytes << " bytes are allocated at " << static_cast<void *>(m_buffer) << std::endl;
}

template <typename T>
bool cpixmap<T>::isMatched(const cpixmap& pixmap) const
{
  const cregion a = static_cast<const cregion>(pixmap);
  return cregion::isMatched(a);
}

template <typename T>
bool cpixmap<T>::isMatched(const cregion& a) const
{
  return cregion::isMatched(a);
}

template <typename T>
bool cpixmap<T>::isMatched(size_t w, size_t h, size_t b) const
{
  return cregion::isMatched(cregion(w, h, b));
}

template <typename T>
inline T *cpixmap<T>::getImage(size_t z) const
{
  return (T *)(m_buffer + z*m_band_stride);
}

template <typename T>
inline T *cpixmap<T>::getLine(size_t y, size_t z) const
{
  return (T *)(m_buffer + z*m_band_stride + y*m_height_stride);
}

template <typename T>
inline T& cpixmap<T>::getPixel(size_t x, size_t y, size_t z) const
{
  return *(T *)(m_buffer + z*m_band_stride + y*m_height_stride + x*sizeof(T));
}

template <typename T>
inline void cpixmap<T>::putPixel(T val, size_t x, size_t y, size_t z)
{
  *(T *)(m_buffer + z*m_band_stride + y*m_height_stride + x*sizeof(T)) = val;
}

template <typename T>
void cpixmap<T>::readVLine(T *line, size_t len, size_t x, size_t y, size_t z) const
{
  uint8_t *p;

  assert(cregion::include(x, y, z));
  
  p = m_buffer + z*m_band_stride + y*m_height_stride + x*sizeof(T);
  for (size_t i = 0; i < std::min(len, m_height-y); ++i) {
    *(line + i) = *(T *)p;
    p += m_height_stride;
  }
}

template <typename T>
void cpixmap<T>::readHLine(T *line, size_t len, size_t x, size_t y, size_t z) const
{
  uint8_t *p;

  assert(cregion::include(x, y, z));
  
  p = m_buffer + z*m_band_stride + y*m_height_stride + x*sizeof(T);
  //#pragma omp parallel for
  for (size_t j = 0; j < std::min(len, m_width-x); ++j) {
    *(line + j) = *((T *)p + j);
  }
}

/*
template <typename T>
cpixmap<T> cpixmap<T>::operator=(const cpixmap<T>& m)
{
  assert(isMatched((cregion)m));
  
}
*/

template <typename T>
void cpixmap<T>::flipHorizontally(void)
{
  for (size_t z = 0; z < m_bands; ++z) {
#pragma omp parallel for
    for (size_t y = 0; y < m_height; ++y) {
      uint8_t *p = m_buffer + z*m_band_stride + y*m_height_stride;
      for (size_t x = 0; x < (m_width>>1); ++x) {
	T temp = *((T *)p + x);
	*((T *)p + x) = *((T *)p + (m_width-1) - x);
	*((T *)p + (m_width-1) - x) = temp;
      }
    }
  }
}

template <typename T>
void cpixmap<T>::flipVertically(void)
{
  for (size_t z = 0; z < m_bands; ++z) {
#pragma omp parallel for
    for (size_t x = 0; x < m_width; ++x) {
      uint8_t *p = m_buffer + z*m_band_stride + x*sizeof(T);
      for (size_t y = 0; y < (m_height>>1); ++y) {
	T temp = *((T *)p + y*m_height_stride);
	*(T *)(p + y*m_height_stride) = *(T *)(p + ((m_height-1) - y)*m_height_stride);
	*(T *)(p + ((m_height-1)-y)*m_height_stride) = temp;
      }
    }
  }
}

template <typename T>
void cpixmap<T>::lshiftPixel(size_t bits)
{
  for (size_t z = 0; z < m_bands; ++z) {
#pragma omp parallel for
    for (size_t y = 0; y < m_height; ++y) {
      T *p = (T *)(m_buffer + z*m_band_stride + y*m_height_stride);
      for (size_t x = 0; x < m_width; ++x) *p++ <<= bits;
    }
  }
}

template <typename T>
void cpixmap<T>::rshiftPixel(size_t bits)
{
  for (size_t z = 0; z < m_bands; ++z) {
#pragma omp parallel for
    for (size_t y = 0; y < m_height; ++y) {
      T *p = (T *)(m_buffer + z*m_band_stride + y*m_height_stride);
      for (size_t x = 0; x < m_width; ++x) *p++ >>= bits;
    }
  }
}
