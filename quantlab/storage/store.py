"""
Experiment storage backend.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from quantlab.storage.experiment import Experiment


logger = logging.getLogger(__name__)


class ExperimentStore:
    """
    Persistent storage for experiments.
    
    Uses SQLite for metadata and JSON files for detailed experiment data.
    """
    
    def __init__(
        self,
        experiments_dir: Path,
        db_path: Optional[Path] = None,
    ):
        """
        Initialize the experiment store.
        
        Args:
            experiments_dir: Directory to store experiment files
            db_path: Path to SQLite database (default: experiments_dir/quantlab.db)
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path or (self.experiments_dir / "quantlab.db")
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_size TEXT,
                quant_method TEXT NOT NULL,
                status TEXT NOT NULL,
                name TEXT,
                tags TEXT,
                timestamp TEXT NOT NULL,
                latency_ms REAL,
                memory_mb REAL,
                throughput_tps REAL,
                file_path TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_name 
            ON experiments(model_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_quant_method 
            ON experiments(quant_method)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON experiments(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def save_experiment(self, experiment: Experiment) -> str:
        """
        Save an experiment to storage.
        
        Args:
            experiment: Experiment to save
            
        Returns:
            Experiment ID
        """
        # Save detailed JSON file
        exp_file = self.experiments_dir / f"{experiment.id}.json"
        with open(exp_file, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2, default=str)
        
        # Update database index
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO experiments (
                id, model_name, model_size, quant_method, status,
                name, tags, timestamp, latency_ms, memory_mb, 
                throughput_tps, file_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment.id,
            experiment.model_name,
            experiment.model_size,
            experiment.quant_method,
            experiment.status,
            experiment.name,
            json.dumps(experiment.tags),
            experiment.timestamp.isoformat() if experiment.timestamp else None,
            experiment.metrics.get("latency_mean_ms"),
            experiment.metrics.get("memory_mb"),
            experiment.metrics.get("throughput_tps"),
            str(exp_file),
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved experiment {experiment.id} to {exp_file}")
        return experiment.id
    
    def load_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Load an experiment by ID.
        
        Args:
            experiment_id: Experiment ID to load
            
        Returns:
            Experiment if found, None otherwise
        """
        exp_file = self.experiments_dir / f"{experiment_id}.json"
        
        if not exp_file.exists():
            logger.warning(f"Experiment file not found: {exp_file}")
            return None
        
        with open(exp_file, "r") as f:
            data = json.load(f)
        
        return Experiment.from_dict(data)
    
    def list_experiments(
        self,
        model_name: Optional[str] = None,
        quant_method: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        limit: int = 20,
        order_by: str = "timestamp",
        descending: bool = True,
    ) -> List[Experiment]:
        """
        List experiments with optional filters.
        
        Args:
            model_name: Filter by model name (partial match)
            quant_method: Filter by quantization method
            tags: Filter by tags (any match)
            status: Filter by status
            limit: Maximum number of results
            order_by: Column to order by
            descending: Order direction
            
        Returns:
            List of matching experiments
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        query = "SELECT id FROM experiments WHERE 1=1"
        params = []
        
        if model_name:
            query += " AND model_name LIKE ?"
            params.append(f"%{model_name}%")
        
        if quant_method:
            query += " AND quant_method = ?"
            params.append(quant_method)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        # Order and limit
        order_dir = "DESC" if descending else "ASC"
        query += f" ORDER BY {order_by} {order_dir} LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Load full experiments
        experiments = []
        for (exp_id,) in rows:
            exp = self.load_experiment(exp_id)
            if exp:
                # Filter by tags if specified
                if tags and not any(t in exp.tags for t in tags):
                    continue
                experiments.append(exp)
        
        return experiments
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.
        
        Args:
            experiment_id: ID of experiment to delete
            
        Returns:
            True if deleted, False if not found
        """
        exp_file = self.experiments_dir / f"{experiment_id}.json"
        
        if not exp_file.exists():
            return False
        
        # Delete file
        exp_file.unlink()
        
        # Delete from database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        conn.commit()
        conn.close()
        
        logger.info(f"Deleted experiment {experiment_id}")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        stats = {}
        
        # Total experiments
        cursor.execute("SELECT COUNT(*) FROM experiments")
        stats["total_experiments"] = cursor.fetchone()[0]
        
        # By quantization method
        cursor.execute("""
            SELECT quant_method, COUNT(*) 
            FROM experiments 
            GROUP BY quant_method
        """)
        stats["by_quant_method"] = dict(cursor.fetchall())
        
        # By model
        cursor.execute("""
            SELECT model_name, COUNT(*) 
            FROM experiments 
            GROUP BY model_name
        """)
        stats["by_model"] = dict(cursor.fetchall())
        
        # By status
        cursor.execute("""
            SELECT status, COUNT(*) 
            FROM experiments 
            GROUP BY status
        """)
        stats["by_status"] = dict(cursor.fetchall())
        
        conn.close()
        return stats
    
    def export_all(self, output_path: Path) -> None:
        """Export all experiments to a single JSON file."""
        experiments = self.list_experiments(limit=10000)
        
        data = {
            "exported_at": datetime.now().isoformat(),
            "count": len(experiments),
            "experiments": [exp.to_dict() for exp in experiments],
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(experiments)} experiments to {output_path}")
    
    def import_from_file(self, input_path: Path) -> int:
        """Import experiments from an export file."""
        with open(input_path, "r") as f:
            data = json.load(f)
        
        count = 0
        for exp_data in data.get("experiments", []):
            exp = Experiment.from_dict(exp_data)
            self.save_experiment(exp)
            count += 1
        
        logger.info(f"Imported {count} experiments from {input_path}")
        return count
