from typing import List, Tuple, Set
import numpy as np
import plotly.graph_objects as go
import copy

class Item:
    def __init__(self, dimensions: Tuple[float, ...]):
        dims = tuple(float(x) for x in dimensions)
        if len(dims) == 1:
            self.original_dim = (dims[0], dims[0], dims[0])
        elif len(dims) == 2:
            self.original_dim = (dims[0], dims[1], dims[1])
        else:
            self.original_dim = dims[:3]
        self.rotations = self.generate_rotations()
        
    def generate_rotations(self) -> List[Tuple[float, float, float]]:
        l, w, h = self.original_dim
        return [
            (l, w, h), (l, h, w),
            (w, l, h), (w, h, l),
            (h, l, w), (h, w, l)
        ]
        
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

class Bin:
    def __init__(self, size: Tuple[float, float, float]):
        self.size = size
        self.items: List[PlacedItem] = []
        self.candidate_positions: Set[Tuple[float, float, float]] = {(0, 0, 0)}
        
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
            
            if abs(top_z - bottom_z) > 1e-5:
                continue
                
            x_overlap = max(0, min(x + l, p_x + p_l) - max(x, p_x))
            y_overlap = max(0, min(y + w, p_y + p_w) - max(y, p_y))
            overlap_area = x_overlap * y_overlap
            
            if overlap_area > 0:
                support_area += overlap_area
                
                for i, (cx, cy) in enumerate(bottom_corners):
                    if (p_x <= cx <= p_x + p_l and p_y <= cy <= p_y + p_w):
                        corner_support[i] = True
        
        if support_area >= min_support * bottom_area - 1e-5:
            return True
            
        if sum(corner_support) >= 3:
            return True
            
        return False
        
    def add_candidate_positions(self, new_item: PlacedItem):
        x, y, z = new_item.corner
        l, w, h = new_item.dimensions
        top_z = z + h
        
        self.candidate_positions.add((x, y, top_z))
        self.candidate_positions.add((x + l, y, top_z))
        self.candidate_positions.add((x, y + w, top_z))
        self.candidate_positions.add((x + l, y + w, top_z))
        self.candidate_positions.add((x + l/2, y + w/2, top_z))
        
    def try_place_item(self, item: Item, min_support: float = 0.7) -> bool:
        for rotation in item.rotations:
            for candidate in sorted(self.candidate_positions, key=lambda c: (c[2], c[0], c[1])):
                x, y, z_base = candidate
                candidate_corner = (x, y, z_base)
                
                if not self.within_bin(candidate_corner, rotation):
                    continue
                    
                overlaps = False
                for placed_item in self.items:
                    if self.overlaps(candidate_corner, rotation, 
                                    placed_item.corner, placed_item.dimensions):
                        overlaps = True
                        break
                if overlaps:
                    continue
                    
                if not self.check_stability(candidate_corner, rotation, min_support):
                    continue
                    
                placed = PlacedItem(candidate_corner, rotation)
                self.items.append(placed)
                self.add_candidate_positions(placed)
                return True
                
        return False
        
    def remaining_volume(self) -> float:
        used = sum(pi.dimensions[0] * pi.dimensions[1] * pi.dimensions[2] 
                  for pi in self.items)
        return self.size[0] * self.size[1] * self.size[2] - used
        
    def get_all_coordinates(self) -> List[List[Tuple[float, float, float]]]:
        return [item.corners() for item in self.items]

def best_fit_bin_packing(
    items_data: List, 
    bin_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_support: float = 0.7,
    default_height: float = 0.1
) -> List[Bin]:
    items = []
    for element in items_data:
        # Extract dimensions - handle different input formats
        if isinstance(element, (tuple, list, np.ndarray)):
            if len(element) == 2:
                # ([centroid], [dimensions]) format
                dim = element[1]
            elif len(element) == 6:
                # [x, y, z, l, w, h] format
                dim = element[3:]
            else:
                # Assume raw dimensions if length doesn't match
                dim = element
        else:
            raise ValueError(f"Invalid item type: {type(element)}")
        
        # Process dimensions
        dim_list = list(dim)
        if len(dim_list) == 1:
            dim_list = [dim_list[0]] * 3
        elif len(dim_list) == 2:
            dim_list.append(default_height)
        elif len(dim_list) > 3:
            dim_list = dim_list[:3]
        
        items.append(Item(tuple(dim_list)))
    
    items.sort(key=lambda item: item.volume, reverse=True)
    bins = []
    
    for item in items:
        best_bin = None
        best_remaining_volume = float('inf')
        original_bin_index = None  # Track which bin we're updating
        
        # Try existing bins
        for idx, bin in enumerate(bins):
            bin_copy = copy.deepcopy(bin)
            if bin_copy.try_place_item(item, min_support):
                remaining_vol = bin_copy.remaining_volume()
                if remaining_vol < best_remaining_volume:
                    best_remaining_volume = remaining_vol
                    best_bin = bin_copy
                    original_bin_index = idx  # Remember original bin position
        
        # Try new bin
        new_bin = Bin(bin_size)
        if new_bin.try_place_item(item, min_support):
            remaining_vol = new_bin.remaining_volume()
            if remaining_vol < best_remaining_volume:
                best_remaining_volume = remaining_vol
                best_bin = new_bin
                original_bin_index = None  # Mark as new bin
        
        # Place item in best bin found
        if best_bin:
            if original_bin_index is not None:
                # Replace existing bin
                bins[original_bin_index] = best_bin
            else:
                # Add new bin
                bins.append(best_bin)
        else:
            print(f"Warning: Could not place item with dimensions {item.original_dim}")
                
    return bins

def visualize_bins(bins: List[Bin]):
    if not bins:
        print("No bins to visualize")
        return
        
    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta',
              'darkblue', 'darkred', 'darkgreen', 'gold', 'violet', 'teal', 'pink']
    
    for bin_idx, bin in enumerate(bins):
        for item_idx, item in enumerate(bin.items):
            corners = item.corners()
            x = [c[0] for c in corners]
            y = [c[1] for c in corners]
            z = [c[2] for c in corners]
            
            i = [0, 0, 0, 0, 7, 4, 4, 6, 4, 0, 3, 6]
            j = [1, 2, 3, 4, 6, 5, 6, 5, 0, 1, 7, 3]
            k = [2, 3, 4, 1, 5, 7, 0, 7, 5, 5, 2, 2]
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color=colors[(bin_idx * len(bin.items) + item_idx) % len(colors)],
                opacity=0.8,
                name=f'Bin {bin_idx+1}-Item {item_idx+1}',
                showlegend=True
            ))
    
    for bin_idx, bin in enumerate(bins):
        x = [0, bin.size[0], bin.size[0], 0, 0, bin.size[0], bin.size[0], 0]
        y = [0, 0, bin.size[1], bin.size[1], 0, 0, bin.size[1], bin.size[1]]
        z = [0, 0, 0, 0, bin.size[2], bin.size[2], bin.size[2], bin.size[2]]
        
        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]
        
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[x[edge[0]], x[edge[1]]],
                y=[y[edge[0]], y[edge[1]]],
                z=[z[edge[0]], z[edge[1]]],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))
    
    max_dim = max(max(bin.size) for bin in bins) if bins else 1
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[0, max_dim]),
            yaxis=dict(title='Y', range=[0, max_dim]),
            zaxis=dict(title='Z', range=[0, max_dim]),
            aspectmode='cube'
        ),
        title='3D Bin Packing Visualization (Best Fit)',
        legend_title="Items per Bin"
    )
    fig.show()

if __name__ == "__main__":
    data = [
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
    
    bins = best_fit_bin_packing(data, bin_size=(1.0, 1.0, 1.0), min_support=0.7)
    
    for i, bin in enumerate(bins):
        print(f"Bin {i+1}:")
        print(f"  Items: {len(bin.items)}")
        print(f"  Remaining volume: {bin.remaining_volume():.4f}")
        for j, item in enumerate(bin.items):
            print(f"  Item {j+1}:")
            print(f"    Corner: {item.corner}")
            print(f"    Dimensions: {item.dimensions}")
            print(f"    Centroid: {item.centroid}")
    
    visualize_bins(bins)