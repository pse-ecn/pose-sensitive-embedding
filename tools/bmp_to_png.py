import argparse
import glob
import os

from PIL import Image


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--directory', help='image directory', dest='image_directory')
	args = parser.parse_args()

	all_bmps = glob.glob(os.path.join(args.image_directory, '**/*.bmp'), recursive=True)

	for i, bmp_path in enumerate(all_bmps):
		image = Image.open(bmp_path)
		png_path = bmp_path[:-3] + 'png'
		image.save(png_path)
		os.remove(bmp_path)

		print('Finished %d: %s' % (i, bmp_path))


if __name__ == '__main__':
	main()
