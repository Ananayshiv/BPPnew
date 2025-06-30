from typing import List, Tuple, Optional, Set, Dict
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict

class Item:
    def __init__(self, dimensions: Tuple[float, ...]):
        # Convert to tuple of floats and ensure exactly 3 dimensions
        dims = tuple(float(x) for x in dimensions)
        if len(dims) == 1:
            self.original_dim = (dims[0], dims[0], dims[0])
        elif len(dims) == 2:
            self.original_dim = (dims[0], dims[1], dims[1])  # Make square base
        else:
            self.original_dim = dims[:3]  # Take first 3 elements
        self.rotations = self.generate_rotations()
        
    def generate_rotations(self) -> List[Tuple[float, float, float]]:
        l, w, h = self.original_dim
        unique_rotations = set()
        
        # Only consider rotations where height is the smallest dimension (for better stacking)
        if h <= l and h <= w:
            unique_rotations.add((l, w, h))
            unique_rotations.add((w, l, h))
        if l <= h and l <= w:
            unique_rotations.add((h, w, l))
            unique_rotations.add((w, h, l))
        if w <= h and w <= l:
            unique_rotations.add((l, h, w))
            unique_rotations.add((h, l, w))
            
        # If all dimensions are equal, just use one rotation
        if l == w == h:
            return [(l, w, h)]
            
        return list(unique_rotations)
        
    @property
    def volume(self) -> float:
        l, w, h = self.original_dim
        return l * w * h

class PlacedItem:
    def __init__(self, corner: Tuple[float, float, float], dimensions: Tuple[float, float, float]):
        self.corner = corner
        self.dimensions = dimensions
        self.centroid = (
            corner[0] + dimensions[0] / 2,
            corner[1] + dimensions[1] / 2,
            corner[2] + dimensions[2] / 2
        )
        
    def corners(self) -> List[Tuple[float, float, float]]:
        x, y, z = self.corner
        l, w, h = self.dimensions
        return [
            (x, y, z),
            (x + l, y, z),
            (x + l, y + w, z),
            (x, y + w, z),
            (x, y, z + h),
            (x + l, y, z + h),
            (x + l, y + w, z + h),
            (x, y + w, z + h)
        ]
    
    def bottom_face(self) -> List[Tuple[float, float, float]]:
        x, y, z = self.corner
        l, w, _ = self.dimensions
        return [
            (x, y, z),
            (x + l, y, z),
            (x + l, y + w, z),
            (x, y + w, z)
        ]
    
    def top_face(self) -> List[Tuple[float, float, float]]:
        x, y, z = self.corner
        l, w, h = self.dimensions
        return [
            (x, y, z + h),
            (x + l, y, z + h),
            (x + l, y + w, z + h),
            (x, y + w, z + h)
        ]

class Bin:
    def __init__(self, size: Tuple[float, float, float]):
        self.size = size
        self.items: List[PlacedItem] = []
        self.candidate_positions: Set[Tuple[float, float, float]] = {(0, 0, 0)}
        self.height_map = defaultdict(float)  # Tracks the highest point at each (x,y) coordinate
        
    @staticmethod
    def overlaps(corner1: Tuple[float, float, float], dim1: Tuple[float, float, float],
                corner2: Tuple[float, float, float], dim2: Tuple[float, float, float]) -> bool:
        x1, y1, z1 = corner1
        l1, w1, h1 = dim1
        x2, y2, z2 = corner2
        l2, w2, h2 = dim2
        
        return not (
            x1 + l1 <= x2 or
            x2 + l2 <= x1 or
            y1 + w1 <= y2 or
            y2 + w2 <= y1 or
            z1 + h1 <= z2 or
            z2 + h2 <= z1
        )
        
    def within_bin(self, corner: Tuple[float, float, float], dims: Tuple[float, float, float]) -> bool:
        x, y, z = corner
        l, w, h = dims
        return (
            x >= 0 and y >= 0 and z >= 0 and
            x + l <= self.size[0] and
            y + w <= self.size[1] and
            z + h <= self.size[2]
        )
        
    def check_stability(self, corner: Tuple[float, float, float], 
                       dims: Tuple[float, float, float], 
                       min_support: float = 0.7) -> bool:
        x, y, z = corner
        l, w, h = dims
        bottom_z = z
        
        # Item is on the floor - stable
        if bottom_z < 1e-5:
            return True
            
        support_area = 0.0
        bottom_area = l * w
        bottom_corners = [
            (x, y), (x + l, y),
            (x, y + w), (x + l, y + w)
        ]
        corner_support = [False] * 4
        
        for placed_item in self.items:
            p_x, p_y, p_z = placed_item.corner
            p_l, p_w, p_h = placed_item.dimensions
            top_z = p_z + p_h
            
            # Check if this item is directly below the candidate
            if abs(top_z - bottom_z) > 1e-5:
                continue
                
            # Calculate overlap area
            x_overlap = max(0, min(x + l, p_x + p_l) - max(x, p_x))
            y_overlap = max(0, min(y + w, p_y + p_w) - max(y, p_y))
            overlap_area = x_overlap * y_overlap
            
            if overlap_area > 0:
                support_area += overlap_area
                
                # Check corner support
                for i, (cx, cy) in enumerate(bottom_corners):
                    if (p_x <= cx <= p_x + p_l and p_y <= cy <= p_y + p_w):
                        corner_support[i] = True
        
        # Check area support
        if support_area >= min_support * bottom_area - 1e-5:
            return True
            
        # Check corner support (3/4 corners supported)
        if sum(corner_support) >= 3:
            return True
            
        return False
        
    def update_height_map(self, new_item: PlacedItem):
        x, y, z = new_item.corner
        l, w, h = new_item.dimensions
        
        # Update height map for the item's footprint
        for xi in np.arange(x, x + l, l/10):
            for yi in np.arange(y, y + w, w/10):
                self.height_map[(round(xi, 2), round(yi, 2))] = max(self.height_map.get((round(xi, 2), round(yi, 2)), 0), z + h)

        
    def get_best_candidate_position(self, dims: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Find the best position considering height and proximity to walls"""
        best_pos = None
        best_score = float('inf')
        
        l, w, h = dims
        
        for candidate in sorted(self.candidate_positions, key=lambda c: (c[2], c[0] + c[1])):
            x, y, z = candidate
            
            # Skip positions that would extend beyond bin boundaries
            if x + l > self.size[0] or y + w > self.size[1]:
                continue
                
            # Calculate a score based on height and distance to walls
            height_score = z
            wall_score = min(x, self.size[0] - (x + l)) + min(y, self.size[1] - (y + w))
            total_score = height_score * 0.7 + wall_score * 0.3
            
            if total_score < best_score:
                best_score = total_score
                best_pos = (x, y, z)
        
        return best_pos if best_pos is not None else (0, 0, 0)
        
    def try_place_item(self, item: Item, min_support: float = 0.7) -> bool:
        # Try all rotations in order of increasing height (better for stacking)
        for rotation in sorted(item.rotations, key=lambda r: r[2]):
            # Get dimensions for this rotation
            l, w, h = rotation
            
            # Find the best candidate position for this rotation
            candidate = self.get_best_candidate_position(rotation)
            if candidate is None:
                continue
                
            x, y, z = candidate
            candidate_corner = (x, y, z)
            
            # Skip invalid positions
            if not self.within_bin(candidate_corner, rotation):
                continue
                
            # Check overlaps with existing items
            overlaps = False
            for placed_item in self.items:
                if self.overlaps(candidate_corner, rotation, 
                                placed_item.corner, placed_item.dimensions):
                    overlaps = True
                    break
            if overlaps:
                continue
                
            # Check stability
            if not self.check_stability(candidate_corner, rotation, min_support):
                continue
                
            # Place the item
            placed = PlacedItem(candidate_corner, rotation)
            self.items.append(placed)
            
            # Update candidate positions
            self.candidate_positions.discard(candidate)
            self.candidate_positions.add((x, y, z + h))  # Top of the new item
            self.candidate_positions.add((x + l, y, z))  # Right side
            self.candidate_positions.add((x, y + w, z))  # Front side
            
            # Update height map
            self.update_height_map(placed)
            
            return True
                
        return False
        
    def remaining_volume(self) -> float:
        used = sum(pi.dimensions[0] * pi.dimensions[1] * pi.dimensions[2] 
                  for pi in self.items)
        return self.size[0] * self.size[1] * self.size[2] - used
        
    def get_all_coordinates(self) -> List[List[Tuple[float, float, float]]]:
        return [item.corners() for item in self.items]

def first_fit_bin_packing(
    items_data: List, 
    bin_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_support: float = 0.7,
    default_height: float = 0.1
) -> List[Bin]:
    bins: List[Bin] = []
    
    # Preprocess items
    items = []
    for element in items_data:
        if isinstance(element, (tuple, list, np.ndarray)):
            if len(element) == 2:
                centroid, dim = element
            elif len(element) == 6:  # Flat array
                centroid, dim = element[:3], element[3:]
            else:
                raise ValueError(f"Invalid item format: {element}")
        else:
            raise ValueError(f"Invalid item type: {type(element)}")
        
        # Convert to list and handle various dimension lengths
        dim_list = list(dim)
        if len(dim_list) == 1:
            dim_list = [dim_list[0]] * 3  # Make cube
        elif len(dim_list) == 2:
            dim_list.append(default_height)
        elif len(dim_list) > 3:
            dim_list = dim_list[:3]  # Take first 3 dimensions
        
        items.append(Item(tuple(dim_list)))
    
    # Sort items by volume (largest first) for better packing
    items.sort(key=lambda x: x.volume, reverse=True)
    
    # Pack items
    for item in items:
        placed = False
        
        # Try existing bins
        for bin in bins:
            if bin.try_place_item(item, min_support):
                placed = True
                break
                
        # Create new bin if needed
        if not placed:
            new_bin = Bin(bin_size)
            if new_bin.try_place_item(item, min_support):
                bins.append(new_bin)
            else:
                print(f"Warning: Could not place item with dimensions {item.original_dim}")
                
    return bins

# Sample usage

if __name__ == "__main__":

    # Sample data with centroids and dimensions
    data = [
        # Format: (centroid, dimensions)
        ([0.35, 0.5, 0.075], [0.4, 0.1, 0.15]),
        ([0.35, 0.625, 0.075], [0.4, 0.15, 0.15]),
        ([0.775, 0.625, 0.25], [0.45, 0.15, 0.5]),
        ([0.35, 0.475, 0.35], [0.4, 0.05, 0.3]),
        ([0.35, 0.6, 0.35], [0.4, 0.2, 0.3]),
        ([0.25, 0.575, 0.175], [0.2, 0.25, 0.05]),
        ([0.075, 0.475, 0.425], [0.15, 0.05, 0.15]),
        ([0.075, 0.6, 0.425], [0.15, 0.2, 0.15]),
        ([0.6, 0.275, 0.25], [0.2, 0.15, 0.5]),
        ([0.6, 0.5, 0.25], [0.2, 0.2, 0.5]),
        ([0.6, 0.775, 0.25], [0.2, 0.15, 0.5]),
        ([0.775, 0.275, 0.25], [0.05, 0.15, 0.5]),
        ([0.25, 0.575, 0.15], [0.15, 0.25, 0.3]),
        ([0.075, 0.575, 0.325], [0.15, 0.25, 0.05])
    ]
    
    bins = first_fit_bin_packing(data, bin_size=(1.0, 1.0, 1.0), min_support=0.7)
    
    # Print results
    for i, bin in enumerate(bins):
        print(f"Bin {i+1}:")
        print(f"  Dimensions: {bin.size}")
        print(f"  Items: {len(bin.items)}")
        print(f"  Packing efficiency: {(1 - bin.remaining_volume() / (bin.size[0]*bin.size[1]*bin.size[2]))*100:.2f}%")
        for j, item in enumerate(bin.items):
            print(f"  Item {j+1}:")
            print(f"    Position: {item.corner}")
            print(f"    Dimensions: {item.dimensions}")
            print(f"    Volume: {item.dimensions[0]*item.dimensions[1]*item.dimensions[2]:.4f}")
    
    visualize_bins(bins)