#include "image2d.h"
#include <string.h>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

#ifdef WIN32
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
#include "stb_image_write.h"
#include "stb_image.h"
#ifdef WIN32
#pragma warning(pop)
#endif


Image2D::Image2D()
{
	m_width = 0;
	m_height = 0;
}

Image2D::Image2D(int w, int h)
{
	resize(w, h);
}

Image2D::Image2D(const Image2D &src, bool flip)
{
	copyFrom(src, flip);
}

Image2D::~Image2D()
{
}

void Image2D::copyFrom(const Image2D &src, bool flip)
{
	m_width = src.m_width;
	m_height = src.m_height;

	if (!flip)
		m_values = src.m_values;
	else
	{
		resize(m_width, m_height);

		for (int y = 0 ; y < m_height ; y++)
			for (int x = 0 ; x < m_width ; x++)
				m_values[x+y*m_width] = src.m_values[(m_width-1-x) + (m_height-1-y)*m_width];
	}
}

void Image2D::resize(int w, int h)
{
	m_width = w;
	m_height = h;

	int s = std::max(w*h, 0);

	m_values.resize(s);
}

bool Image2D::importPicture(const std::string &fileName, bool flipX, bool flipY)
{
	int w, h, c;
	unsigned char* data = stbi_load(fileName.c_str(), &w, &h, &c, 3);

	if (data == 0)
	{
		setErrorString("Can't load file " + fileName + ": " + std::string(stbi_failure_reason()));
		return false;
	}

	m_width = w;
	m_height = h;

	m_values.resize(m_width*m_height);

	for (int y = 0 ; y < m_height ; y++)
	{
		for (int x = 0 ; x < m_width ; x++)
		{
			int X = x;
			int Y = m_height-1-y;

			if (flipX)
				X = m_width-1-x;
			if (flipY)
				Y = y;
			
			unsigned char* thisdata = data + (3 * (X + Y * m_width));
			m_values[x+y*m_width] = (float)((thisdata[0] * 11 + thisdata[1] * 16 + thisdata[2] * 5)/32);
		}
	}
	
	stbi_image_free(data);

	return true;
}

bool Image2D::exportPicture(const std::string &fileName, bool flipX, bool flipY, bool autoScale) const
{
	if (m_width <= 0 || m_height <= 0)
	{
		setErrorString("No valid image has been loaded or created yet");
		return false;
	}

	std::vector<float> scaledValues;
	const float *pValues = 0;

	if (autoScale)
	{
		scaledValues.resize(m_values.size());
		
		float maxValue = m_values[0];
		float minValue = m_values[0];

		int total = (int)m_values.size();

		for (int i = 0 ; i < total ; i++)
		{
			float v = m_values[i];

			maxValue = std::max(v, maxValue);
			minValue = std::min(v, minValue);
		}

		float rescale = 255.0f/(maxValue-minValue);

		for (int i = 0 ; i < total ; i++)
		{
			float v = m_values[i];

			v -= minValue;
			v *= rescale;

			scaledValues[i] = v;
		}

		pValues = &(scaledValues[0]);
	}
	else
		pValues = &(m_values[0]);
	
	unsigned char* rawimage = new unsigned char[m_width * m_height * 3];

	for (int y = 0 ; y < m_height ; y++)
	{
		for (int x = 0 ; x < m_width ; x++)
		{
			int X = x;
			int Y = m_height-1-y;

			if (flipX)
				X = m_width-1-x;
			if (flipY)
				Y = y;

			float v = pValues[x+y*m_width];

			v = std::min(255.0f, std::max(v, 0.0f));

			int intValue = (int)(v+0.5f);
			
			for(int c = 0; c < 3; ++c)
				rawimage[c + X * 3 + Y * m_width * 3] = intValue;
		}
	}

	if (stbi_write_png((fileName + std::string(".png")).c_str(), m_width, m_height, 3, rawimage, 0) == 0)
	{
		setErrorString("Can't export to file " + fileName);
		delete[] rawimage;
		return false;
	}
	
	delete[] rawimage;

	return true;
}

