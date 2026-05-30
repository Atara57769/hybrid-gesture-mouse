class CoordinateMapper:
    """
    Implements Active Zone interpolation mapping and Exponential Moving Average (EMA) smoothing
    for stable cursor tracking without raw camera coordinate jitter.
    """
    def __init__(self, screen_width: int, screen_height: int, smoothing: float = 0.25, active_zone_pct: float = 0.6):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.smoothing = smoothing
        self.active_zone_pct = active_zone_pct
        self.prev_x = None
        self.prev_y = None
        
    def reset(self, x: int, y: int) -> None:
        """Seeds the coordinate mapper state to prevent massive initial coordinate jumps."""
        self.prev_x = x
        self.prev_y = y
        
    def map_and_smooth(self, raw_x: float, raw_y: float, frame_width: int, frame_height: int) -> tuple[int, int]:
        """
        Maps frame coordinates (raw pixel coordinates on the camera stream) to display coordinates
        scaled relative to the central active zone, then applies EMA smoothing.
        """
        # Calculate boundaries of the Active Zone
        margin_x = (1.0 - self.active_zone_pct) / 2.0
        margin_y = (1.0 - self.active_zone_pct) / 2.0
        
        az_x_min = int(frame_width * margin_x)
        az_x_max = int(frame_width * (1.0 - margin_x))
        az_y_min = int(frame_height * margin_y)
        az_y_max = int(frame_height * (1.0 - margin_y))
        
        # Calculate percentages within the Active Zone
        if az_x_max == az_x_min:
            x_pct = 0.5
        else:
            x_pct = (raw_x - az_x_min) / (az_x_max - az_x_min)
            
        if az_y_max == az_y_min:
            y_pct = 0.5
        else:
            y_pct = (raw_y - az_y_min) / (az_y_max - az_y_min)
            
        # Bound percentage between [0.0, 1.0]
        x_pct = max(0.0, min(1.0, x_pct))
        y_pct = max(0.0, min(1.0, y_pct))
        
        # Linearly interpolate to screen bounds
        target_x = int(x_pct * self.screen_width)
        target_y = int(y_pct * self.screen_height)
        
        # Fallback if reset wasn't called
        if self.prev_x is None or self.prev_y is None:
            self.prev_x = target_x
            self.prev_y = target_y
            
        # Exponential Moving Average (EMA)
        smoothed_x = int(self.smoothing * target_x + (1.0 - self.smoothing) * self.prev_x)
        smoothed_y = int(self.smoothing * target_y + (1.0 - self.smoothing) * self.prev_y)
        
        self.prev_x = smoothed_x
        self.prev_y = smoothed_y
        
        return smoothed_x, smoothed_y
