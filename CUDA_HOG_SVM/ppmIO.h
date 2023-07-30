/*
 * ppmIO.h
 *
 *  Created on: Sep 2, 2020
 *      Author: vision
 */

#ifndef PPMIO_H_
#define PPMIO_H_

#include <fstream>

inline void getPPMSize(const char *filename, int *width, int *height)
{
	char buff[16];
	FILE *fp;
	int c;

	// open PPM file for reading
	fp = fopen(filename, "rb");
	if (!fp)
	{
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	// read image format
	if (!fgets(buff, sizeof(buff), fp))
	{
		perror(filename);
		exit(1);
	}

	// check the image format
	if (buff[0] != 'P' || buff[1] != '6')
	{
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	// check for comments
	c = getc(fp);
	while (c == '#')
	{
		while (getc(fp) != '\n')
			;
		c = getc(fp);
	}

	ungetc(c, fp);
	// read image size information
	if (fscanf(fp, "%d %d", &(*width), &(*height)) != 2)
	{
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}
}

inline void readPPM(const char *filename, float *image)
{
	char buff[16];
	FILE *fp;
	int c, rgb_comp_color;
	unsigned int width, height;

	// open PPM file for reading
	fp = fopen(filename, "rb");
	if (!fp)
	{
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	// read image format
	if (!fgets(buff, sizeof(buff), fp))
	{
		perror(filename);
		exit(1);
	}

	// check the image format
	if (buff[0] != 'P' || buff[1] != '6')
	{
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	// check for comments
	c = getc(fp);
	while (c == '#')
	{
		while (getc(fp) != '\n')
			;
		c = getc(fp);
	}

	ungetc(c, fp);
	// read image size information
	if (fscanf(fp, "%d %d", &width, &height) != 2)
	{
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}

	// read rgb component
	if (fscanf(fp, "%d", &rgb_comp_color) != 1)
	{
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
		exit(1);
	}

	// check rgb component depth
	if (rgb_comp_color != 255)
	{
		fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
		exit(1);
	}

	while (fgetc(fp) != '\n')
		;

	// allocate temporary memory for unsigned char data
	unsigned char *tempImage = (unsigned char *)malloc(width * height * 3 * sizeof(unsigned char));

	// read pixel data from file
	if (fread(tempImage, sizeof(unsigned char), 3 * width * height, fp) != 3 * width * height)
	{
		fprintf(stderr, "Error loading image '%s'\n", filename);
		exit(1);
	}

	// transfer data from unsigned char to float
	int i;
	for (i = 0; i < width * height * 3; ++i)
	{
		image[i] = (float)tempImage[i];
	}

	// cleanup
	free(tempImage);
	fclose(fp);
}

void writePPM(const char *filename, float *image, unsigned int width, unsigned int height, bool isGray = 0)
{
	FILE *fp;
	unsigned int channels = isGray ? 1 : 3;
	// open file for output
	fp = fopen(filename, "wb");
	if (!fp)
	{
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	// write the header file
	// image format
	fprintf(fp, "P6\n");

	// comments
	fprintf(fp, "# Created for CUDA labs AGH\n");

	// image size
	fprintf(fp, "%d %d\n", width, height);

	// rgb component depth
	fprintf(fp, "%d\n", 255);

	// copy from float* to uchar*
	unsigned char *tempImage = (unsigned char *)malloc(width * height * 3 * sizeof(unsigned char));
	int i, j = 0;
	for (i = 0; i < width * height * channels; ++i)
	{

		j = isGray ? j : i;
		tempImage[j] = (unsigned char)image[i];

		if (isGray)
		{
			tempImage[j + 1] = (unsigned char)image[i];
			tempImage[j + 2] = (unsigned char)image[i];
			j += 3;
		}
	}

	// pixel data
	fwrite(tempImage, sizeof(unsigned char), 3 * width * height, fp);
	free(tempImage);
	fclose(fp);
}

#endif /* PPMIO_H_ */
