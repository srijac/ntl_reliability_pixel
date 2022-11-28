import numpy as np
import pickle
from dotenv import load_dotenv
from os import environ
from pathlib import Path
import json

# Load environmental variables
load_dotenv()


class PolyInfoDict:

    __slots__: ["dictionary"]

    def __init__(self):
        print('----- IN POLYINFODICT----------')
        # Load polygon information dictionary
        with open(Path(environ["support_files_path"], "poly_info.json"), encoding="utf-8") as f:
            poly_info = json.load(f)
        # Reference
        self.dictionary = poly_info


class PolyTilePixelDict:

    __slots__: ["dictionary"]

    def __init__(self):
        print('----- IN POLYTILEPIXELDICT----------')
        # Load polygon > tile > pixel > area dictionary
        with open(Path(environ["support_files_path"], "poly_to_tile_to_pixel.json")) as f:
            poly_to_tile_to_pixel = json.load(f)
        # Reference dictionary
        self.dictionary = poly_to_tile_to_pixel


class TilePolyDict:

    __slots__: ["dictionary"]

    def __init__(self):
        print('----- IN POLYTILEPIXELDICT 2----------')
        # Load tile > polygon dictionary
        with open(Path(environ["support_files_path"], "tile_to_poly.json")) as f:
            tile_to_poly = json.load(f)
        # Reference the loaded dictionary
        self.dictionary = tile_to_poly


class BoundingBox:

    __slots__: ["west_long",
                "east_long",
                "north_lat",
                "south_lat",
                "tile_dict"]

    def __init__(self):

        self.west_long = None
        self.east_long = None
        self.north_lat = None
        self.south_lat = None
        self.tile_dict = {}

    # Define the bounding box with Center point lat long (Coordinate objects), width and height
    def define_center_width_height(self, center_lat, center_long, width, height):
        # If the width is degrees and decimal minutes
        if len(width) == 2:
            # Convert to decimal degrees
            width = ddm_to_decdegs(width[0], width[1])
        # Otherwise, if the width is degrees, minutes and seconds
        elif len(width) == 3:
            # Convert to decimal degrees
            width = dms_to_decdegs(width[0], width[1], width[2])
        # If the height is degrees and decimal minutes
        if len(height) == 2:
            # Convert to decimal degrees
            height = ddm_to_decdegs(height[0], height[1])
        # Otherwise, if the height is degrees, minutes and seconds
        elif len(height) == 3:
            # Convert to decimal degrees
            height = dms_to_decdegs(height[0], height[1], height[2])
        # Make coordinate objects of the center latitude and longitude (converts to decimal degrees)
        lat_co = Coordinate(center_lat)
        long_co = Coordinate(center_long)
        # Derive the boundaries
        self.west_long = Coordinate(lat_co.dec_degs - (width / 2))
        self.east_long = Coordinate(lat_co.dec_degs + (width / 2))
        self.north_lat = Coordinate(long_co.dec_degs + (height / 2))
        self.south_lat = Coordinate(long_co.dec_degs - (height / 2))

        # Get the tile and pixel extents in VNP46A grid
        self.get_tiles_pixels()

    # Define the bounding box with Upper Left lat long and Lower Right lat long (Coordinate objects)
    def define_ul_lr(self, ul_lat, ul_long, lr_lat, lr_long):
        # Transfer the boundaries
        self.west_long = Coordinate(ul_lat.dec_degs)
        self.east_long = Coordinate(lr_lat.dec_degs)
        self.north_lat = Coordinate(ul_long.dec_degs)
        self.south_lat = Coordinate(lr_long.dec_degs)

        # Get the tile and pixel extents in VNP46A grid
        self.get_tiles_pixels()

    # Define the bounding box with East and West lats, and North and South longs
    def define_north_south_east_west(self, north, south, east, west):
        # Transfer the boundaries
        self.west_long = west
        self.east_long = east
        self.north_lat = north
        self.south_lat = south

        # Get the tile and pixel extents in VNP46A grid
        self.get_tiles_pixels()

    # Retrieve the tiles and pixels for this bounding box
    def get_tiles_pixels(self):
        # Transfer the coordinates
        west_tile, west_pixel = get_tile_pixel_from_long(self.west_long)
        east_tile, east_pixel = get_tile_pixel_from_long(self.east_long, east_boundary=True)
        north_tile, north_pixel = get_tile_pixel_from_lat(self.north_lat)
        south_tile, south_pixel = get_tile_pixel_from_lat(self.south_lat, south_boundary=True)

        # Get tile horizontal numbers
        tile_hs = np.arange(west_tile, east_tile + 1)
        # Get tile vertical numbers
        tile_vs = np.arange(north_tile, south_tile + 1)
        # Start the pixel index arrays
        horizontal_pixels = [west_pixel]
        vertical_pixels = [north_pixel]
        # For each horizontal tile
        for h_count, tile_h in enumerate(tile_hs):
            # If this is not the first tile
            if h_count != 0:
                # Add indices for the horizontal pixels
                horizontal_pixels.extend([2399, 0])
            # For each vertical tile
            for v_count, tile_v in enumerate(tile_vs):
                # If this is the first h tile and not the first v tile
                if h_count == 0 and v_count != 0:
                    # Add indices for the vertical pixels
                    vertical_pixels.extend([2399, 0])
                # Assemble the string zero-padded h and v coordinates
                str_h = str(tile_h)
                if len(str_h) < 2:
                    str_h = '0' + str_h
                str_v = str(tile_v)
                if len(str_v) < 2:
                    str_v = '0' + str_v
                # Assemble the tile name
                tile_str = 'h' + str_h + 'v' + str_v
                # Add the tile name to the dictionary
                self.tile_dict[tile_str] = {}
        # Add the finishing indices for the pixels
        horizontal_pixels.append(east_pixel)
        vertical_pixels.append(south_pixel)
        horizontal_pixels = np.reshape(np.array(horizontal_pixels), (int(len(horizontal_pixels) / 2), 2))
        vertical_pixels = np.reshape(np.array(vertical_pixels), (int(len(vertical_pixels) / 2), 2))

        # For each horizontal tile
        for h_count, tile_h in enumerate(tile_hs):
            # For each vertical tile
            for v_count, tile_v in enumerate(tile_vs):
                # Assemble the string zero-padded h and v coordinates
                str_h = str(tile_h)
                if len(str_h) < 2:
                    str_h = '0' + str_h
                str_v = str(tile_v)
                if len(str_v) < 2:
                    str_v = '0' + str_v
                # Assemble the tile name
                tile_str = 'h' + str_h + 'v' + str_v
                # Add the tile name to the dictionary
                self.tile_dict[tile_str] = {"pixel_col_min": horizontal_pixels[h_count][0],
                                            "pixel_col_max": horizontal_pixels[h_count][1],
                                            "pixel_row_min": vertical_pixels[v_count][0],
                                            "pixel_row_max": vertical_pixels[v_count][1]}


class Coordinate:

    __slots__: ["dec_degs"]

    def __init__(self, coordinate):

        # If the format is Degrees and Decimal Minutes (ddm or DDM)
        if len(coordinate) == 2:
            # Convert to decimal degrees
            self.dec_degs = ddm_to_decdegs(coordinate[0], coordinate[1])
        # Otherwise, if the format is Degrees, Minutes, Seconds (dms)
        elif len(coordinate) == 3:
            # Convert to decimal degrees
            self.dec_degs = dms_to_decdegs(coordinate[0], coordinate[1], coordinate[2])
        # Otherwise, it's decimal degrees already
        else:
            self.dec_degs = coordinate[0]


# Convert degrees, minutes, seconds to decimal degrees
def dms_to_decdegs(degrees, minutes, seconds):
    # Return the decimal degrees
    return degrees + (minutes / 60) + (seconds / 3600)


# Convert degrees and decimal minutes to decimal degrees
def ddm_to_decdegs(degrees, dec_mins):
    # Return the decimal degrees
    return degrees + (dec_mins / 60)


# Get a VNP46 grid tile and pixel from a latitude (Coordinate object)
def get_tile_pixel_from_lat(coordinate, south_boundary=False):
    # Tiles run from 0-17 for +90 to -90 latitude
    # Tiles run from 0 - 43199 pixels
    tile_number = 18 - ((coordinate.dec_degs + 90) / 10)
    # If the tile number is right on a boundary
    if tile_number % 1 == 0:
        # If it's a South boundary
        if south_boundary is True:
            # Subtract 1 from the tile number
            tile_number -= 1
    print(tile_number)
    # Get the pixel number
    pixel_number = (tile_number % 1) / (1 / 2400)
    # If the pixel number is right on a boundary
    if pixel_number % 1 == 0:
        # If it's a South boundary
        if south_boundary is True:
            # Subtract 1 from the pixel number
            pixel_number -= 1
            # If the pixel number is -1 (previous tile)
            if pixel_number == -1:
                # Change to 2399 (last in previous tile)
                pixel_number = 2399
    return int(np.floor(tile_number)), int(np.floor(pixel_number))


# Get a VNP46 grid tile and pixel from a longitude (Coordinate object)
def get_tile_pixel_from_long(coordinate, east_boundary=False):
    # Tiles run from 0-35 for -180 to +180 longitude
    tile_number = ((coordinate.dec_degs + 180) / 10)
    # If the tile number is right on a boundary
    if tile_number % 1 == 0:
        # If it's an East boundary
        if east_boundary is True:
            # Subtract 1 from the tile number
            tile_number -= 1
    print(tile_number)
    # Get the pixel number
    pixel_number = (tile_number % 1) / (1 / 2400)
    print(pixel_number)
    # If the pixel number is right on a boundary
    if pixel_number % 1 == 0:
        # If it's an East boundary
        if east_boundary is True:
            # Subtract 1 from the pixel number
            pixel_number -= 1
            # If the pixel number is -1 (previous tile)
            if pixel_number == -1:
                # Change to 2399 (last in previous tile)
                pixel_number = 2399
    return int(np.floor(tile_number)), int(np.floor(pixel_number))


# For calculating top and bottom lengths for pixel (calgary.rasc.ca/latlong.htm for equations)
def longitude_distance(latitude):
    # Convert latitude to radians
    latitude = np.radians(latitude)
    # Calculate degrees of longitude for one pixel
    long_degs_per_pixel = 360 / 86400
    # Return longitudinal distance given latitude
    return long_degs_per_pixel * ((111.41288 * (np.cos(latitude))) - (0.09350 * (np.cos(3 * latitude))) + (
            0.00012 * (np.cos(5 * latitude))))


# For calculating height of pixel (calgary.rasc.ca/latlong.htm for equations)
def latitude_distance(latitude):
    # Convert latitude to radians
    latitude = np.radians(latitude)
    # Calculate degrees of latitude for one pixel
    lat_degs_per_pixel = 180 / 43200
    # Return latitudinal distance given latitude
    return lat_degs_per_pixel * (111.13295 - (0.55982 * np.cos(2 * latitude)) + (0.00117 * np.cos(4 * latitude)))


# Pixel latitudes (top and bottom of pixel) given the y coordinate of the pixel (where 0 is at the top)
def pixel_lat(y_co):
    # Degrees per pixel
    lat_degs_per_pixel = 180 / 43200
    # Pixel top and bottom latitudes
    pixel_top_lat = (abs(21600 - y_co)) * lat_degs_per_pixel
    pixel_bottom_lat = (abs(21600 - (y_co + 1))) * lat_degs_per_pixel
    # Return top and bottom lats
    return pixel_top_lat, pixel_bottom_lat


# Derive true area of pixel given y coordinate (0 is top "row" of the world, 21,600 is the equator)
def get_pixel_area(global_y):
    # Retrieve pixel top and bottom latitudes
    pixel_top_lat, pixel_bottom_lat = pixel_lat(global_y)
    # Retrieve pixel top and bottom distances
    pixel_top_dist = longitude_distance(pixel_top_lat)
    pixel_bottom_dist = longitude_distance(pixel_bottom_lat)
    # Retrieve pixel "height" distance based on top latitude
    pixel_height = latitude_distance(pixel_top_lat)
    # Return area as trapezoid approximation in km^2
    return ((pixel_top_dist + pixel_bottom_dist) / 2) * pixel_height