import os
import pickle
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

class CheckpointSystem:
    def __init__(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            # Set default checkpoint directory to the results/pipeline_run folder
            project_root = Path(__file__).parent.parent
            self.checkpoint_dir = project_root / 'results' / 'pipeline_run'
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create the checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Checkpoint directory: {self.checkpoint_dir}")
    
    def _save_dataframe(self, df, file_path):
        """Save DataFrame in parquet format for better performance"""
        try:
            # Use parquet for better compression and speed
            parquet_path = file_path.with_suffix('.parquet')
            df.to_parquet(parquet_path, compression='snappy')
            return parquet_path
        except Exception:
            # Fallback to pickle if parquet fails
            with open(file_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(df, f)
            return file_path.with_suffix('.pkl')
    
    def _load_dataframe(self, file_path):
        """Load DataFrame from parquet or pickle"""
        parquet_path = file_path.with_suffix('.parquet')
        pickle_path = file_path.with_suffix('.pkl')
        
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        elif pickle_path.exists():
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Neither parquet nor pickle file found for {file_path.stem}")
    
    def save_checkpoint(self, data, step_name, description=""):
        """Save checkpoint data with optimized handling for DataFrames"""
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Handle different data types
            if isinstance(data, pd.DataFrame):
                # Save DataFrame directly
                data_file = self._save_dataframe(data, self.checkpoint_dir / step_name)
                data_info = {
                    'type': 'dataframe',
                    'shape': data.shape,
                    'memory_usage': data.memory_usage(deep=True).sum(),
                    'columns': list(data.columns)
                }
            elif isinstance(data, dict):
                # Handle dictionary that might contain DataFrames
                data_file = self.checkpoint_dir / f"{step_name}.pkl"
                data_info = {'type': 'dict', 'keys': list(data.keys())}
                
                # Check for DataFrames in the dictionary
                df_info = {}
                modified_data = {}
                
                for key, value in data.items():
                    if isinstance(value, pd.DataFrame):
                        # Save DataFrame separately
                        df_file = self._save_dataframe(value, self.checkpoint_dir / f"{step_name}_{key}")
                        df_info[key] = {
                            'file': str(df_file.name),
                            'shape': value.shape,
                            'memory_usage': value.memory_usage(deep=True).sum()
                        }
                        # Store reference instead of actual DataFrame
                        modified_data[key] = f"__DATAFRAME_REF__{key}"
                    else:
                        modified_data[key] = value
                
                # Save the modified dictionary
                with open(data_file, 'wb') as f:
                    pickle.dump(modified_data, f)
                
                if df_info:
                    data_info['dataframes'] = df_info
            else:
                # Default pickle for other data types
                data_file = self.checkpoint_dir / f"{step_name}.pkl"
                with open(data_file, 'wb') as f:
                    pickle.dump(data, f)
                data_info = {'type': type(data).__name__}
            
            # Save metadata
            metadata = {
                'step_name': step_name,
                'description': description,
                'timestamp': datetime.now().isoformat(),
                'file_size': os.path.getsize(data_file),
                'data_info': data_info
            }
            
            metadata_file = self.checkpoint_dir / f"{step_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"✅ Checkpoint saved: {step_name}")
            print(f"   📄 Data: {data_file}")
            if 'dataframes' in data_info:
                for df_name, df_info in data_info['dataframes'].items():
                    print(f"   🗃️  DataFrame '{df_name}': {df_info['shape']} shape")
            
        except Exception as e:
            print(f"❌ Error saving checkpoint {step_name}: {str(e)}")
            raise
    
    def load_checkpoint(self, step_name):
        """Load checkpoint data with optimized DataFrame handling"""
        try:
            metadata_file = self.checkpoint_dir / f"{step_name}_metadata.json"
            metadata = {}
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            data_info = metadata.get('data_info', {})
            
            if data_info.get('type') == 'dataframe':
                # Load DataFrame directly
                data = self._load_dataframe(self.checkpoint_dir / step_name)
            elif data_info.get('type') == 'dict' and 'dataframes' in data_info:
                # Load dictionary with DataFrames
                data_file = self.checkpoint_dir / f"{step_name}.pkl"
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Restore DataFrames
                for df_name, df_info in data_info['dataframes'].items():
                    if data.get(df_name) == f"__DATAFRAME_REF__{df_name}":
                        df_file = self.checkpoint_dir / df_info['file']
                        data[df_name] = self._load_dataframe(df_file.with_suffix(''))
            else:
                # Default pickle loading
                data_file = self.checkpoint_dir / f"{step_name}.pkl"
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
            
            print(f"✅ Checkpoint loaded: {step_name}")
            if metadata:
                print(f"   📅 Saved: {metadata.get('timestamp', 'Unknown')}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading checkpoint {step_name}: {str(e)}")
            raise
    
    def list_checkpoints(self):
        """List all available checkpoints"""
        checkpoints = []
        
        for pkl_file in self.checkpoint_dir.glob("*.pkl"):
            step_name = pkl_file.stem
            
            # Skip metadata files
            if step_name.endswith('_metadata'):
                continue
            
            metadata_file = self.checkpoint_dir / f"{step_name}_metadata.json"
            metadata = {}
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass
            
            checkpoints.append({
                'step_name': step_name,
                'file_path': str(pkl_file),
                'file_size': pkl_file.stat().st_size,
                'timestamp': metadata.get('timestamp', 'Unknown'),
                'description': metadata.get('description', 'No description')
            })
        
        return sorted(checkpoints, key=lambda x: x['timestamp'])
    
    def checkpoint_exists(self, step_name):
        """Check if a checkpoint exists"""
        data_file = self.checkpoint_dir / f"{step_name}.pkl"
        return data_file.exists()
    
    def delete_checkpoint(self, step_name):
        """Delete a checkpoint and its metadata"""
        try:
            data_file = self.checkpoint_dir / f"{step_name}.pkl"
            metadata_file = self.checkpoint_dir / f"{step_name}_metadata.json"
            
            if data_file.exists():
                data_file.unlink()
                print(f"🗑️  Deleted checkpoint data: {step_name}")
            
            if metadata_file.exists():
                metadata_file.unlink()
                print(f"🗑️  Deleted checkpoint metadata: {step_name}")
                
        except Exception as e:
            print(f"❌ Error deleting checkpoint {step_name}: {str(e)}")
            raise
    
    def get_checkpoint_info(self, step_name):
        """Get information about a specific checkpoint"""
        if not self.checkpoint_exists(step_name):
            return None
        
        data_file = self.checkpoint_dir / f"{step_name}.pkl"
        metadata_file = self.checkpoint_dir / f"{step_name}_metadata.json"
        
        info = {
            'step_name': step_name,
            'data_file': str(data_file),
            'file_size': data_file.stat().st_size,
            'exists': True
        }
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                info.update(metadata)
            except:
                pass
        
        return info
    
    def _get_data_summary(self, data):
        """Get a brief summary of the data"""
        if data is None:
            return "None"
        elif isinstance(data, pd.DataFrame):
            return f"DataFrame: {data.shape}"
        elif isinstance(data, np.ndarray):
            return f"Array: {data.shape}"
        elif hasattr(data, 'shape'):
            return f"shape: {data.shape}"
        elif isinstance(data, (list, tuple)):
            return f"length: {len(data)}"
        elif isinstance(data, dict):
            return f"keys: {len(data)}"
        else:
            return f"type: {type(data).__name__}"

    def print_summary(self):
        """Print a summary of all checkpoints and pipeline execution"""
        print(f"\n📊 CHECKPOINT SUMMARY - Run ID: {getattr(self, 'run_id', 'unknown')}")
        print("=" * 60)
        
        # Calculate total execution time if available
        total_time = getattr(self, 'total_execution_time', None)
        if total_time:
            minutes = int(total_time // 60)
            seconds = total_time % 60
            print(f"⏱️  Total pipeline time: {total_time:.2f}s ({minutes}m {seconds:.1f}s)")
        
        print(f"📁 Checkpoint directory: {self.checkpoint_dir}")
        
        # List all checkpoints
        checkpoints = self.list_checkpoints()
        print(f"💾 Total checkpoints: {len(checkpoints)}")
        
        if checkpoints:
            print(f"\n📋 Step Details:")
            print("-" * 75)
            
            for checkpoint in checkpoints:
                step_name = checkpoint['step_name']
                timestamp = checkpoint.get('timestamp', 'Unknown')
                description = checkpoint.get('description', 'No description')
                file_size = checkpoint.get('file_size', 0)
                
                # Format file size
                if file_size > 1024*1024:
                    size_str = f"{file_size/(1024*1024):.1f}MB"
                elif file_size > 1024:
                    size_str = f"{file_size/1024:.1f}KB"
                else:
                    size_str = f"{file_size}B"
                
                print(f"📦 {step_name}")
                print(f"   📅 Time: {timestamp}")
                print(f"   📝 Description: {description}")
                print(f"   💾 Size: {size_str}")
                
                # Try to load and summarize data (safely)
                try:
                    data = self.load_checkpoint(step_name)
                    summary = self._get_data_summary(data)
                    print(f"   📊 Data: {summary}")
                except Exception as e:
                    print(f"   ❌ Error loading: {str(e)[:50]}...")
                
                print()
        else:
            print("\n📭 No checkpoints found")

    def get_total_execution_time(self):
        """Get total execution time if available"""
        if hasattr(self, 'start_time') and hasattr(self, 'end_time'):
            return (self.end_time - self.start_time).total_seconds()
        elif hasattr(self, 'total_execution_time'):
            return self.total_execution_time
        else:
            return 0.0