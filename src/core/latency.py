import time
from typing import Dict, List
import numpy as np

class LatencyLogger:
    def __init__(self):
        self.metrics = {
            "stt": [],
            "embedding": [],
            "retrieval": [],
            "generation": [],
            "end_to_end": []
        }
        
    def log(self, stage: str, duration_ms: float):
        """Log latency for a specific stage in milliseconds."""
        if stage in self.metrics:
            self.metrics[stage].append(duration_ms)
        else:
            self.metrics[stage] = [duration_ms]
            
    def get_percentiles(self, stage: str) -> Dict[str, float]:
        """Calculate P50, P70, P100 for a given stage."""
        if not self.metrics.get(stage):
            return {"p50": 0.0, "p70": 0.0, "p100": 0.0, "count": 0}
            
        data = self.metrics[stage]
        return {
            "p50": float(np.percentile(data, 50)),
            "p70": float(np.percentile(data, 70)),
            "p100": float(np.percentile(data, 100)),
            "count": len(data)
        }
        
    def report(self):
        """Print a formatted latency report."""
        print("\n" + "="*50)
        print(" LATENCY REPORT (ms)")
        print("="*50)
        print(f"{'STAGE':<12} | {'P50':<6} | {'P70':<6} | {'P100':<6} | {'COUNT':<5}")
        print("-" * 50)
        for stage in self.metrics:
            stats = self.get_percentiles(stage)
            if stats.get("count", 0) > 0:
                print(f"{stage.upper():<12} | {stats['p50']:6.1f} | {stats['p70']:6.1f} | {stats['p100']:6.1f} | {stats['count']:<5}")
        print("="*50 + "\n")

# Global logger instance for convenience
logger = LatencyLogger()
