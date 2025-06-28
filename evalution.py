import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

class ConvexHullAnalyzer:
    def __init__(self, bin_size=(1.0, 1.0, 1.0)):
        self.bin_size = np.array(bin_size)
        self.items = []
    
    def add_item(self, corners):
        """Add an item represented by its 8 corners (list of 3D points)"""
        self.items.append(np.array(corners))
    
    def reset(self):
        """Clear all items"""
        self.items = []
    
    def compute_convex_hull_volume(self):
        """Compute convex hull volume of all items"""
        if not self.items:
            return 0.0
            
        all_points = np.vstack(self.items)
        hull = ConvexHull(all_points)
        return hull.volume

    def compute_bounding_box_volume(self):
        """Compute axis-aligned bounding box volume"""
        if not self.items:
            return 0.0
            
        all_points = np.vstack(self.items)
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)
        return np.prod(max_vals - min_vals)

    def compute_total_item_volume(self):
        """Compute sum of all individual item volumes"""
        total = 0.0
        for item in self.items:
            # Calculate item dimensions from its corners
            min_vals = np.min(item, axis=0)
            max_vals = np.max(item, axis=0)
            total += np.prod(max_vals - min_vals)
        return total

    def compute_metrics(self):
        """Compute all packing efficiency metrics"""
        hull_vol = self.compute_convex_hull_volume()
        item_vol = self.compute_total_item_volume()
        bbox_vol = self.compute_bounding_box_volume()
        
        deviation = hull_vol - item_vol
        hull_ratio = hull_vol / bbox_vol if bbox_vol > 0 else 0
        efficiency = (item_vol / hull_vol) * 100 if hull_vol > 0 else 0
        
        return {
            'convex_hull_volume': hull_vol,
            'item_volume': item_vol,
            'bounding_box_volume': bbox_vol,
            'empty_space': deviation,
            'hull_to_bbox_ratio': hull_ratio,
            'packing_efficiency': efficiency
        }

    def visualize(self):
        """Visualize items and convex hull"""
        if not self.items:
            print("No items to visualize!")
            return
            
        metrics = self.compute_metrics()
        all_points = np.vstack(self.items)
        hull = ConvexHull(all_points)
        
        fig = go.Figure()
        
        # Plot convex hull
        fig.add_trace(go.Mesh3d(
            x=all_points[:, 0], y=all_points[:, 1], z=all_points[:, 2],
            i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
            color='lightpink', opacity=0.6, name='Convex Hull'
        ))
        
        # Plot bin boundaries
        X, Y, Z = self.bin_size
        bin_edges = [
            [0,0,0], [X,0,0], [X,Y,0], [0,Y,0], [0,0,0],
            [0,0,Z], [X,0,Z], [X,Y,Z], [0,Y,Z], [0,0,Z],
            [X,0,Z], [X,0,0], [X,Y,0], [X,Y,Z], [0,Y,Z], [0,Y,0]
        ]
        bin_edges = np.array(bin_edges)
        fig.add_trace(go.Scatter3d(
            x=bin_edges[:, 0], y=bin_edges[:, 1], z=bin_edges[:, 2],
            mode='lines', line=dict(color='black', width=3), name='Bin'
        ))
        
        # Plot items
        for i, item in enumerate(self.items):
            fig.add_trace(go.Scatter3d(
                x=item[:, 0], y=item[:, 1], z=item[:, 2],
                mode='markers', marker=dict(size=3, color='blue'),
                name=f'Item {i+1}'
            ))
        
        # Set plot layout
        fig.update_layout(
            title=f"3D Bin Packing - Efficiency: {metrics['packing_efficiency']:.1f}%",
            scene=dict(
                xaxis=dict(range=[0, self.bin_size[0]]),
                yaxis=dict(range=[0, self.bin_size[1]]),
                zaxis=dict(range=[0, self.bin_size[2]]),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.show()
        
        # Print metrics
        print("\nPACKING METRICS:")
        print(f"• Convex Hull Volume: {metrics['convex_hull_volume']:.4f}")
        print(f"• Total Item Volume: {metrics['item_volume']:.4f}")
        print(f"• Bounding Box Volume: {metrics['bounding_box_volume']:.4f}")
        print(f"• Empty Space: {metrics['empty_space']:.4f}")
        print(f"• Hull/BBox Ratio: {metrics['hull_to_bbox_ratio']:.4f}")
        print(f"• Packing Efficiency: {metrics['packing_efficiency']:.1f}%")
        
        return metrics


# Example Usage
if __name__ == "__main__":
    # Initialize analyzer with bin size
    analyzer = ConvexHullAnalyzer(bin_size=(3, 3, 3))
    
    # Add items (each defined by 8 corner points)
    item1=[(0, 0, 0), (0.35, 0, 0), (0.35, 0.625, 0), (0, 0.625, 0), (0, 0, 0.075), (0.35, 0, 0.075), (0.35, 0.625, 0.075), (0, 0.625, 0.075)]
    item2 = [
        [1,0,0], [2,0,0], [2,1,0], [1,1,0],
        [1,0,1], [2,0,1], [2,1,1], [1,1,1]
    ]
    
    analyzer.add_item(item1)
    analyzer.add_item(item2)
    
    # Get metrics without visualization
    metrics = analyzer.compute_metrics()
    print("\nQuick Volume Check:")
    print(f"Convex Hull Volume: {metrics['convex_hull_volume']:.4f}")
    print(f"Hull/BBox Ratio: {metrics['hull_to_bbox_ratio']:.4f}")
    
    # Visualize and show full metrics
    analyzer.visualize()