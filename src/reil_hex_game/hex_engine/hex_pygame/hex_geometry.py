# Credits: https://github.com/parappayo/hex-py/blob/master/game_geometry.py
# License: (MIT License) https://github.com/parappayo/hex-py/blob/master/LICENSE
# Author: Parappayo
# Modified by: HackXIt

import math, itertools
from functools import lru_cache
from typing import List


def distance_squared(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x1 - x2, y2 - y1
    return dx * dx + dy * dy


def points_up_tile_size_px(size):
    return math.floor(size * math.sqrt(3)), size * 2


def flats_up_tile_size_px(size):
    return size * 2, math.floor(size * math.sqrt(3))


@lru_cache(maxsize=128)
def points_up_tile_center_point(grid_position, width, height, offset):
    x, y = grid_position
    dx, dy = offset
    height = math.floor(height * 3/4)

    # stagger odd rows
    if y % 2:
        dx += width // 2

    # diamond-shaped grid
    x += y // 2

    return (x * width + dx, y * height + dy)


@lru_cache(maxsize=128)
def flats_up_tile_center_point(grid_position, width, height, offset):
    x, y = grid_position
    dx, dy = offset
    width = math.floor(width * 3/4)

    # stagger odd columns
    if x % 2:
        dy += height // 2

    # diamond-shaped grid
    y += x // 2

    return (x * width + dx, y * height + dy)


def points_up_tile_corner_point(radius, index, position_px):
    theta = math.tau * index / 6 + math.tau / 12
    x, y = position_px
    return (radius * math.cos(theta) + x, radius * math.sin(theta) + y)


def flats_up_tile_corner_point(radius, index, position_px):
    theta = math.tau * index / 6
    x, y = position_px
    return (radius * math.cos(theta) + x, radius * math.sin(theta) + y)


@lru_cache(maxsize=128)
def points_up_tile_corner_points(grid_position, width, height, offset):
    radius = height // 2
    position_px = points_up_tile_center_point(grid_position, width, height, offset)
    return [points_up_tile_corner_point(radius, i, position_px) for i in range(6)]


@lru_cache(maxsize=128)
def flats_up_tile_corner_points(grid_position, width, height, offset):
    radius = width // 2
    position_px = flats_up_tile_center_point(grid_position, width, height, offset)
    return [flats_up_tile_corner_point(radius, i, position_px) for i in range(6)]


class HexTile:

    def __init__(self, grid_x, grid_y, size_px, points_up):
        self.grid_position = (grid_x, grid_y)
        self.neighbours: List[HexTile] = []
        if points_up:
            self.width, self.height = points_up_tile_size_px(size_px)
        else:
            self.width, self.height = flats_up_tile_size_px(size_px)
        self.points_up = points_up


    def __str__(self):
        return f'{self.grid_position}'


    def __repr__(self):
        return f'HexTile{self.grid_position}'


    def center_point(self, offset=0):
        y, x = self.grid_position
        rotated_grid = (x, y)
        if self.points_up:
            return points_up_tile_center_point(
                rotated_grid,
                self.width,
                self.height,
                offset)
        else:
            return flats_up_tile_center_point(
                rotated_grid,
                self.width,
                self.height,
                offset)


    def corner_points(self, offset=0):
        y, x = self.grid_position
        rotated_grid = (x, y)
        if self.points_up:
            return points_up_tile_corner_points(
                rotated_grid,
                self.width,
                self.height,
                offset)
        else:
            return flats_up_tile_corner_points(
                rotated_grid,
                self.width,
                self.height,
                offset)


    def distance_squared(self, position, offset):
        return distance_squared(self.center_point(offset), position)


class HexGrid:

    def __init__(self, width, height, tile_size, points_up):
        self.width = width
        self.height = height
        self.tiles = {
            (x,y): HexTile(x, y, tile_size, points_up)
            for (x,y) in itertools.product(range(width), range(height)) }
        for tile in self.tiles.values():
            self.populate_neighbours(tile)


    def populate_neighbours(self, tile):
        x, y = tile.grid_position
        if x > 0:
            tile.neighbours.append(self.tiles[(x-1, y)])
        if x < self.width-1:
            tile.neighbours.append(self.tiles[(x+1, y)])
        if y > 0:
            tile.neighbours.append(self.tiles[(x, y-1)])
            if x < self.width-1:
                tile.neighbours.append(self.tiles[(x+1, y-1)])
        if y < self.height-1:
            tile.neighbours.append(self.tiles[(x, y+1)])
            if x > 0:
                tile.neighbours.append(self.tiles[(x-1, y+1)])


    def find_path(self, from_tile, to_tiles, filter, visited=None):
        if visited == None:
            visited = []

        if not filter(from_tile) or from_tile in visited:
            return None

        if from_tile in to_tiles:
            return [from_tile]

        visited.append(from_tile)

        for neighbour in from_tile.neighbours:
            result = self.find_path(neighbour, to_tiles, filter, visited)
            if result != None:
                result.append(from_tile)
                return result

        return None


    def top_row(self):
        return [self.tiles[(x, 0)] for x in range(self.width)]


    def bottom_row(self):
        return [self.tiles[(x, self.height-1)] for x in range(self.width)]


    def left_column(self):
        return [self.tiles[(0, y)] for y in range(self.height)]


    def right_column(self):
        return [self.tiles[(self.width-1, y)] for y in range(self.height)]