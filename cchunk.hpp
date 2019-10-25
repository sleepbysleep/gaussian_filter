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
#include "cpixmap.hpp"

// so called a tile of image
template <typename T>
class cchunk {
public:
  cchunk(void);
  cchunk(size_t width, size_t height, size_t hpadding, size_t vpadding);
  virtual ~cchunk(void);
  void setDimension(size_t width, size_t height, size_t hpadding, size_t vpadding);
  void draft(const cpixmap<T>& image, size_t x = 0, size_t y = 0, size_t z = 0);
  void shiftByNextLines(size_t lines_to_read, const cpixmap<T>& image, size_t z = 0);
  T& operator() (int y, int x);
private:
  void reallocate(size_t lines, size_t stride);
  size_t m_width;
  size_t m_height;
  size_t m_horizontal_padding;
  size_t m_vertical_padding;
  size_t m_stride;
  int m_horizontal_start;
  int m_vertical_start;
  uint8_t *m_buffer;
  T **m_line_buffer;
};

template <typename T>
cchunk<T>::cchunk(void)
  : m_width(0),
    m_height(0),
    m_horizontal_padding(0),
    m_vertical_padding(0),
    m_stride(0),
    m_horizontal_start(0),
    m_vertical_start(0),
    m_buffer(NULL),
    m_line_buffer(NULL) {}

template <typename T>
cchunk<T>::cchunk(size_t width, size_t height, size_t hpadding, size_t vpadding)
  : m_width(0),
    m_height(0),
    m_horizontal_padding(0),
    m_vertical_padding(0),
    m_stride(0),
    m_horizontal_start(0),
    m_vertical_start(0),
    m_buffer(NULL),
    m_line_buffer(NULL)
{
  setDimension(width, height, hpadding, vpadding);
}

template <typename T>
cchunk<T>::~cchunk(void)
{
  if (m_buffer) delete [] m_buffer;
  if (m_line_buffer) delete [] m_line_buffer;
}

template <typename T>
void cchunk<T>::setDimension(size_t width, size_t height, size_t hpadding, size_t vpadding)
{
  m_width = width;
  m_height = height;
  m_horizontal_padding = hpadding;
  m_vertical_padding = vpadding;
  m_stride = QWORD_ALIGN((width + (hpadding<<1)) * sizeof(T));

  reallocate(height + (vpadding<<1), m_stride);
}

template <typename T>
void cchunk<T>::draft(const cpixmap<T>& image, size_t x, size_t y, size_t z)
{
  //assert(m_stride == QWORD_ALIGN((image.getWidth()+(m_horizontal_padding<<1))*sizeof(T)));
  assert(m_buffer);
  assert(m_line_buffer);

  m_horizontal_start = x - m_horizontal_padding;
  m_vertical_start = y - m_vertical_padding;
    
  size_t lines = m_height + (m_vertical_padding<<1);    
  std::memset(m_buffer, 0, lines * m_stride);
    
  for (size_t i = 0; i < lines; i++)
    m_line_buffer[i] = (T *)(m_buffer + i*m_stride);

  size_t hoffset = std::max(m_horizontal_start, 0) - m_horizontal_start;
  size_t voffset = std::max(m_vertical_start, 0) - m_vertical_start;

  for (size_t i = voffset; i < lines; i++) {
    image.readHLine(m_line_buffer[i] + hoffset,
		    m_width + (m_horizontal_padding<<1) - hoffset,
		    m_horizontal_start+hoffset,
		    m_vertical_start+i,
		    z);
  }
}

template <typename T>
void cchunk<T>::shiftByNextLines(size_t lines_to_read, const cpixmap<T>& image, size_t z)
{
  //assert(m_stride == QWORD_ALIGN((image.getWidth()+(m_horizontal_padding<<1))*sizeof(T)));
  assert(m_buffer);
  assert(m_line_buffer);
    
  size_t lines_allocated = m_height + (m_vertical_padding<<1);
    
  assert(lines_to_read <= lines_allocated);

  T *temp[lines_to_read];
  for (size_t i = 0; i < lines_to_read; i++)
    temp[i] = m_line_buffer[i];

  for (size_t i = lines_to_read; i < lines_allocated; i++)
    m_line_buffer[i - lines_to_read] = m_line_buffer[i];

  for (size_t i = 0; i < lines_to_read; i++)
    m_line_buffer[lines_allocated - lines_to_read + i] = temp[i];

  m_vertical_start += lines_to_read;

  size_t hoffset = std::max(m_horizontal_start, 0) - m_horizontal_start;
     
  for (size_t i = 0; i < lines_to_read; i++) {
    size_t voffset = lines_allocated - lines_to_read + i;
    size_t line = (size_t)(m_vertical_start + voffset);
    if (line < image.getHeight()) {
      image.readHLine(m_line_buffer[voffset] + hoffset,
		      m_width + (m_horizontal_padding<<1) - hoffset,
		      m_horizontal_start + hoffset,
		      m_vertical_start + voffset,
		      z);
    } else {
      std::memset(m_line_buffer[voffset], 0, m_stride);
    }
  }
}

template <typename T>
T& cchunk<T>::operator()(int y, int x)
{
  assert(y >= m_vertical_start && y < m_vertical_start + (int)(m_height + (m_vertical_padding<<1)));
  assert(x >= m_horizontal_start && x < m_horizontal_start + (int)(m_width + (m_horizontal_padding<<1)));
    
  return *(m_line_buffer[y-m_vertical_start] + x-m_horizontal_start);
}

template <typename T>
void cchunk<T>::reallocate(size_t lines, size_t stride)
{
  if (m_buffer) delete [] m_buffer;
  if (m_line_buffer) delete [] m_line_buffer;
  m_buffer = new uint8_t[lines * stride];
  m_line_buffer = new T*[lines];
}

template <typename T>
class cslice {
public:
  cslice(void) : m_base(NULL) { m_base = new cchunk<T>; }
  cslice(const cpixmap<T>& img, size_t lines, size_t hpadding, size_t vpadding)
    : m_base(NULL)
  {
    m_base = new cchunk<T>;
    m_base->setDimension(img.getWidth(), lines, hpadding, vpadding);
  }
  virtual ~cslice(void) { delete m_base; }
  void setSlice(const cpixmap<T>& img, size_t lines, size_t hpadding, size_t vpadding)
  {
    m_base->setDimension(img.getWidth(), lines, hpadding, vpadding);
  }
  void draftSlice(const cpixmap<T>& img, size_t z = 0) { m_base->draft(img, 0, 0, z); }
  void shiftSlice(size_t lines_to_read, const cpixmap<T>& img, size_t z = 0)
  {
    m_base->shiftByNextLines(lines_to_read, img, z);
  }
  T& operator()(int y, int x) { return (*m_base)(y, x); }
private:
  cchunk<T> *m_base;
};

template <typename T>
class window3x3_frame {
public:
  window3x3_frame(void) : m_base(NULL) { m_base = new cchunk<T>; }
  window3x3_frame(const cpixmap<T>& img)
    : m_base(NULL)
  {
    m_base = new cchunk<T>;
    m_base->setDimension(img.getWidth(), 1, 1, 1);
  }
  virtual ~window3x3_frame(void) { delete m_base; }
  void setFrame(const cpixmap<T>& img) { m_base->setDimension(img.getWidth(), 1, 1, 1); }
  void draftFrame(const cpixmap<T>& img, size_t z = 0) { m_base->draft(img, 0, 0, z); }
  void shiftFrame(const cpixmap<T>& img, size_t z = 0) { m_base->shiftByNextLines(1, img, z); }
  T& operator() (int y, int x) { return (*m_base)(y, x); }
private:
  cchunk<T> *m_base;
};

template <typename T>
class window5x5_frame {
public:
  window5x5_frame(void) : m_base(NULL) { m_base = new cchunk<T>; }
  window5x5_frame(const cpixmap<T>& img)
    : m_base(NULL)
  {
    m_base = new cchunk<T>;
    m_base->setDimension(img.getWidth(), 1, 2, 2);
  }
  virtual ~window5x5_frame(void) { delete m_base; }
  void setFrame(const cpixmap<T>& img) { m_base->setDimension(img.getWidth(), 1, 2, 2); }
  void draftFrame(const cpixmap<T>& img, size_t z = 0) { m_base->draft(img, 0, 0, z); }
  void shiftFrame(const cpixmap<T>& img, size_t z = 0) { m_base->shiftByNextLines(1, img, z); }
  T& operator() (int y, int x) { return (*m_base)(y, x); }
private:
  cchunk<T> *m_base;
};
