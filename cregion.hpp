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

template <typename T>
class cpoint {
public:
  cpoint(T x = 0, T y = 0, T z = 0) : m_x(x), m_y(y), m_z(z) {}
  cpoint(cpoint<T>& point) : m_x(point.getX()), m_y(point.getY()), m_z(point.getZ()) {}
  virtual ~cpoint() {}
  inline T getX(void) const { return m_x; }
  inline T getY(void) const { return m_y; }
  inline T getZ(void) const { return m_z; }
  inline void setX(const T x) { m_x = x; }
  inline void setY(const T y) { m_y = y; }
  inline void setZ(const T z) { m_z = z; }
  inline void setPoint(const cpoint<T>& point) { m_x = point.getX(), m_y = point.getY(), m_z = point.getZ(); }
  cpoint<T>& operator+(const cpoint<T>& rhs);
  cpoint<T>& operator+=(const cpoint<T>& rhs);
  cpoint<T>& operator-(const cpoint<T>& rhs);
  cpoint<T>& operator-=(const cpoint<T>& rhs);
  // prefix increment operator
  cpoint<T>& operator++(void);
  // postfix increment operator
  cpoint<T> operator++(int);

private:
  T m_x, m_y, m_z;
};

template <typename T>
cpoint<T>& cpoint<T>::operator+(const cpoint<T>& rhs)
{
  cpoint<T> temp(m_x + rhs.getX(), m_y + rhs.getY(), m_z + rhs.getZ());
  return temp;
}

template <typename T>
cpoint<T>& cpoint<T>::operator+=(const cpoint<T>& rhs)
{
  m_x += rhs.getX(), m_y += rhs.getY(), m_z += rhs.getZ();
  return *this;
}

template <typename T>
cpoint<T>& cpoint<T>::operator-(const cpoint<T>& rhs)
{
  cpoint<T> temp(m_x - rhs.getX(), m_y - rhs.getY(), m_z - rhs.getZ());
  return temp;
}

template <typename T>
cpoint<T>& cpoint<T>::operator-=(const cpoint<T>& rhs)
{
  m_x -= rhs.getX(), m_y -= rhs.getY(), m_z += rhs.getZ();
  return *this;
}

// prefix increment operator
template <typename T>
cpoint<T>& cpoint<T>::operator++(void)
{
  ++m_x, ++m_y, ++m_z;
  return *this;
}

// postfix increment operator
template <typename T>
cpoint<T> cpoint<T>::operator++(int)
{
  cpoint<T> temp(*this);
  this->operator++();
  return temp;
}

template <typename T>
class cline {
public:
  cline()
    : m_begin(cpoint<T>::cpoint()), m_end(cpoint<T>::cpoint()) {}
  cline(const cpoint<T>& begin, const cpoint<T>& end)
    : m_begin(begin), m_end(end) {}
  virtual ~cline() {}
private:
  cpoint<T> m_begin, m_end;
};

template <typename T>
class ctriangle {
  ctriangle()
    : m_a(cpoint<T>::cpoint()), m_b(cpoint<T>::cpoint()), m_c(cpoint<T>::cpoint()) {}
  ctriangle(const cpoint<T>& a, const cpoint<T>& b, const cpoint<T>& c)
    : m_a(a), m_b(b), m_c(c) {}
private:
  cpoint<T> m_a, m_b, m_c;
};

template <typename T>
class cregion {
public:
  cregion()
    : m_x(0), m_y(0), m_z(0), m_width(0), m_height(0), m_bands(0) {}
  cregion(T w, T h, T b = 1)
    : m_x(0), m_y(0), m_z(0), m_width(w), m_height(h), m_bands(b) {}
  cregion(T x, T y, T w, T h)
    : m_x(x), m_y(y), m_z(0), m_width(w), m_height(h), m_bands(1) {}
  cregion(T x, T y, T z, T w, T h, T b = 1)
    : m_width(w), m_height(h), m_bands(b), m_x(x), m_y(y), m_z(z) {}
  virtual ~cregion() { }
  virtual void setResolution(T w, T h, T b = 1);
  T getWidth(void) const;
  T getHeight(void) const;
  T getBands(void) const;
  void setOrigin(T x, T y, T z);
  T getXOrigin(void) const;
  T getYOrigin(void) const;
  T getZOrigin(void) const;
  T getXEnd(void) const;
  T getYEnd(void) const;
  T getZEnd(void) const;

  bool include(const T x, const T y, const T z = 0) const;
  bool include(const cpoint<T>& pt) const;
  virtual bool isMatched(const cregion& dim) const;
  int getLeftHalf(void) const;
  int getRightHalf(void) const;
  int getUpHalf(void) const;
  int getDownHalf(void) const;
protected:
  T m_x, m_y, m_z;  
  T m_width, m_height; // x x y
  T m_bands; // z
private:
};

template <typename T>
inline bool cregion<T>::include(const T x, const T y, const T z) const
{
  //  return (x >= 0 && x < m_width && y >= 0 && y < m_height && z >= 0 && z < m_bands);
  return (x >= m_x && x < (m_x+m_width) && y >= m_y && y < (m_y+m_height) && z >= m_z && z < (m_z+m_bands));
}

template <typename T>
inline bool cregion<T>::include(const cpoint<T>& pt) const
{
  return (pt.getX() >= m_x && pt.getX() < (m_x+m_width) &&
	  pt.getY() >= m_y && pt.getY() < (m_y+m_height) &&
	  pt.getZ() >= m_z && pt.getZ() < (m_z+m_bands));
}

template <typename T>
inline bool cregion<T>::isMatched(const cregion& dim) const
{
  //  return m_width == dim.m_width && m_height == dim.m_height && m_bands == dim.m_bands;
  return m_x == dim.m_x && m_width == dim.m_width &&
    m_y == dim.m_y && m_height == dim.m_height &&
    m_z == dim.m_z && m_bands == dim.m_bands;
}

template <typename T>
inline void cregion<T>::setResolution(T w, T h, T b)
{
  m_width = w, m_height = h, m_bands = b;
}

template <typename T>
inline T cregion<T>::getWidth(void) const { return m_width; }
template <typename T>
inline T cregion<T>::getHeight(void) const { return m_height; }
template <typename T>
inline T cregion<T>::getBands(void) const { return m_bands; }

template <typename T>
inline void cregion<T>::setOrigin(T x, T y, T z) { m_x = x, m_y = y, m_z = z; }
template <typename T>
inline T cregion<T>::getXOrigin(void) const { return m_x; }
template <typename T>
inline T cregion<T>::getYOrigin(void) const { return m_y; }
template <typename T>
inline T cregion<T>::getZOrigin(void) const { return m_z; }
template <typename T>
inline T cregion<T>::getXEnd(void) const { return m_x + m_width; }
template <typename T>
inline T cregion<T>::getYEnd(void) const { return m_y + m_height; }
template <typename T>
inline T cregion<T>::getZEnd(void) const { return m_z + m_bands; }

template <typename T>
inline int cregion<T>::getLeftHalf(void) const
{
  return (m_width>>1) + 1 - m_width; // negative value
}

template <typename T>
inline int cregion<T>::getRightHalf(void) const
{
  return (m_width>>1) + 1; // positive value
}

template <typename T>
inline int cregion<T>::getUpHalf(void) const
{
  return (m_height>>1) + 1 - m_height; // negative value
}

template <typename T>
inline int cregion<T>::getDownHalf(void) const
{
  return (m_height>>1) + 1;
}
