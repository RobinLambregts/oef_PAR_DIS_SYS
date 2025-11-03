#ifndef IMAGE2D_H

#define IMAGE2D_H

#include <vector>
#include <string>

#define IMAGE2D_SAVEFILE_IDLEN								6

class Image2D
{
public:
	Image2D();
	Image2D(int w, int h);
	Image2D(const Image2D &src, bool flip = false);
	~Image2D();

	std::string getErrorString() const					{ return m_errorString; }

	void resize(int w, int h);
	int getWidth() const								{ return m_width; }
	int getHeight() const								{ return m_height; }

	void copyFrom(const Image2D &src, bool flip = false);

	bool importPicture(const std::string &fileName, bool flipX = false, bool flipY = false);
	bool exportPicture(const std::string &fileName, bool flipX = false, bool flipY = false, bool autoScale = true) const;

	const float *getValues() const						{ return &(m_values[0]); }
	float *getBufferPointer()							{ return &(m_values[0]); }
private:
	void setErrorString(const std::string &str) const	{ m_errorString = str; }

	mutable std::string m_errorString;

	std::vector<float> m_values;
	int m_width, m_height, m_channels;

	static char m_saveFileIdentifier[IMAGE2D_SAVEFILE_IDLEN];
};

#endif // IMAGE2D_H
