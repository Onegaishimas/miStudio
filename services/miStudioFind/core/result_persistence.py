# core/result_persistence.py
"""
Enhanced file persistence for miStudioFind results.
Extends existing ResultManager with comprehensive file storage.
"""

import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List
import torch
import pandas as pd
from datetime import datetime

class EnhancedResultPersistence:
    """Enhanced result persistence with multiple format support."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.results_base = self.data_path / "results"
        self.results_base.mkdir(exist_ok=True)
    
    def save_comprehensive_results(self, job_id: str, results: Dict[str, Any]) -> Dict[str, str]:
        """Save results in all supported formats."""
        job_dir = self.results_base / job_id
        job_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        
        # 1. JSON format (detailed, API-compatible)
        json_path = job_dir / f"{job_id}_complete_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        saved_files['json'] = str(json_path)
        
        # 2. CSV format (spreadsheet-friendly)
        csv_path = job_dir / f"{job_id}_feature_analysis.csv"
        self._save_csv_format(results, csv_path)
        saved_files['csv'] = str(csv_path)
        
        # 3. XML format (structured data exchange)
        xml_path = job_dir / f"{job_id}_analysis.xml"
        self._save_xml_format(results, xml_path)
        saved_files['xml'] = str(xml_path)
        
        # 4. PyTorch format (for downstream processing)
        pt_path = job_dir / f"{job_id}_features.pt"
        self._save_pytorch_format(results, pt_path)
        saved_files['pytorch'] = str(pt_path)
        
        # 5. Summary report (human-readable)
        report_path = job_dir / f"{job_id}_summary_report.txt"
        self._save_summary_report(results, report_path)
        saved_files['summary'] = str(report_path)
        
        # 6. Metadata index
        index_path = job_dir / f"{job_id}_index.json"
        self._save_index_file(job_id, results, saved_files, index_path)
        saved_files['index'] = str(index_path)
        
        return saved_files
    
    def _save_csv_format(self, results: Dict[str, Any], csv_path: Path):
        """Save results in CSV format for spreadsheet analysis."""
        feature_data = []
        
        if 'results' in results:
            for feature in results['results']:
                # Main feature data
                row = {
                    'feature_id': feature.get('feature_id'),
                    'coherence_score': feature.get('coherence_score', 0.0),
                    'quality_level': feature.get('quality_level'),
                    'pattern_keywords': ', '.join(feature.get('pattern_keywords', [])),
                    'mean_activation': feature.get('statistics', {}).get('mean_activation', 0.0),
                    'max_activation': feature.get('statistics', {}).get('max_activation', 0.0),
                    'activation_frequency': feature.get('statistics', {}).get('activation_frequency', 0.0),
                    'top_activation_count': len(feature.get('top_activations', [])),
                }
                
                # Add top activating texts (first 3)
                top_texts = feature.get('top_activations', [])[:3]
                for i, activation in enumerate(top_texts):
                    row[f'top_text_{i+1}'] = activation.get('text', '')[:100] + '...' if len(activation.get('text', '')) > 100 else activation.get('text', '')
                    row[f'top_activation_{i+1}'] = activation.get('activation_value', 0.0)
                
                feature_data.append(row)
        
        df = pd.DataFrame(feature_data)
        df.to_csv(csv_path, index=False)
    
    def _save_xml_format(self, results: Dict[str, Any], xml_path: Path):
        """Save results in XML format for structured data exchange."""
        root = ET.Element("miStudioFindResults")
        
        # Metadata
        metadata = ET.SubElement(root, "metadata")
        ET.SubElement(metadata, "jobId").text = results.get('job_id', '')
        ET.SubElement(metadata, "sourceJobId").text = results.get('source_job_id', '')
        ET.SubElement(metadata, "timestamp").text = datetime.now().isoformat()
        ET.SubElement(metadata, "totalFeatures").text = str(len(results.get('results', [])))
        
        # Features
        features_elem = ET.SubElement(root, "features")
        
        if 'results' in results:
            for feature in results['results']:
                feature_elem = ET.SubElement(features_elem, "feature")
                feature_elem.set("id", str(feature.get('feature_id')))
                
                # Basic properties
                ET.SubElement(feature_elem, "coherenceScore").text = str(feature.get('coherence_score', 0.0))
                ET.SubElement(feature_elem, "qualityLevel").text = feature.get('quality_level', '')
                
                # Keywords
                keywords_elem = ET.SubElement(feature_elem, "patternKeywords")
                for keyword in feature.get('pattern_keywords', []):
                    ET.SubElement(keywords_elem, "keyword").text = keyword
                
                # Top activations
                activations_elem = ET.SubElement(feature_elem, "topActivations")
                for activation in feature.get('top_activations', [])[:5]:  # Top 5
                    act_elem = ET.SubElement(activations_elem, "activation")
                    act_elem.set("rank", str(activation.get('ranking', 0)))
                    ET.SubElement(act_elem, "text").text = activation.get('text', '')
                    ET.SubElement(act_elem, "value").text = str(activation.get('activation_value', 0.0))
        
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    
    def _save_pytorch_format(self, results: Dict[str, Any], pt_path: Path):
        """Save in PyTorch format for downstream processing."""
        pytorch_data = {
            'job_metadata': {
                'job_id': results.get('job_id'),
                'source_job_id': results.get('source_job_id'),
                'total_features': len(results.get('results', [])),
                'timestamp': datetime.now().isoformat()
            },
            'feature_analysis': {}
        }
        
        if 'results' in results:
            for feature in results['results']:
                feature_id = feature.get('feature_id')
                pytorch_data['feature_analysis'][feature_id] = {
                    'coherence_score': feature.get('coherence_score', 0.0),
                    'quality_level': feature.get('quality_level'),
                    'pattern_keywords': feature.get('pattern_keywords', []),
                    'statistics': feature.get('statistics', {}),
                    'top_activations': feature.get('top_activations', [])[:10]  # Top 10
                }
        
        torch.save(pytorch_data, pt_path)
    
    def _save_summary_report(self, results: Dict[str, Any], report_path: Path):
        """Generate human-readable summary report."""
        with open(report_path, 'w') as f:
            f.write("miStudioFind Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Job information
            f.write(f"Job ID: {results.get('job_id', 'N/A')}\n")
            f.write(f"Source Job: {results.get('source_job_id', 'N/A')}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing Time: {results.get('processing_time', 0):.1f} seconds\n\n")
            
            # Summary statistics
            feature_results = results.get('results', [])
            if feature_results:
                quality_dist = {'high': 0, 'medium': 0, 'low': 0}
                coherence_scores = []
                
                for feature in feature_results:
                    quality = feature.get('quality_level', 'low')
                    quality_dist[quality] += 1
                    coherence_scores.append(feature.get('coherence_score', 0.0))
                
                f.write("Analysis Summary:\n")
                f.write(f"Total Features Analyzed: {len(feature_results)}\n")
                f.write(f"High Quality Features: {quality_dist['high']}\n")
                f.write(f"Medium Quality Features: {quality_dist['medium']}\n")
                f.write(f"Low Quality Features: {quality_dist['low']}\n")
                f.write(f"Mean Coherence Score: {sum(coherence_scores)/len(coherence_scores):.3f}\n\n")
                
                # Top features
                sorted_features = sorted(feature_results, key=lambda x: x.get('coherence_score', 0), reverse=True)
                f.write("Top 10 Most Coherent Features:\n")
                f.write("-" * 30 + "\n")
                
                for i, feature in enumerate(sorted_features[:10], 1):
                    f.write(f"{i:2d}. Feature {feature.get('feature_id', 'N/A'):3d} ")
                    f.write(f"(coherence: {feature.get('coherence_score', 0):.3f}) - ")
                    f.write(f"Keywords: {', '.join(feature.get('pattern_keywords', [])[:3])}\n")
    
    def _save_index_file(self, job_id: str, results: Dict[str, Any], saved_files: Dict[str, str], index_path: Path):
        """Save index file for easy result access."""
        index_data = {
            'job_id': job_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_features': len(results.get('results', [])),
            'processing_time_seconds': results.get('processing_time', 0),
            'available_formats': list(saved_files.keys()),
            'file_paths': saved_files,
            'quick_stats': {
                'high_quality_features': len([f for f in results.get('results', []) if f.get('quality_level') == 'high']),
                'mean_coherence': sum(f.get('coherence_score', 0) for f in results.get('results', [])) / max(1, len(results.get('results', []))),
                'interpretable_features': len([f for f in results.get('results', []) if f.get('coherence_score', 0) >= 0.7])
            }
        }
        
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def load_results(self, job_id: str, format: str = 'json') -> Any:
        """Load results in specified format."""
        job_dir = self.results_base / job_id
        
        if format == 'json':
            json_path = job_dir / f"{job_id}_complete_results.json"
            with open(json_path, 'r') as f:
                return json.load(f)
        
        elif format == 'csv':
            csv_path = job_dir / f"{job_id}_feature_analysis.csv"
            return pd.read_csv(csv_path)
        
        elif format == 'pytorch':
            pt_path = job_dir / f"{job_id}_features.pt"
            return torch.load(pt_path)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def list_available_results(self) -> List[Dict[str, Any]]:
        """List all available result sets."""
        available = []
        
        for job_dir in self.results_base.iterdir():
            if job_dir.is_dir():
                index_file = job_dir / f"{job_dir.name}_index.json"
                if index_file.exists():
                    with open(index_file, 'r') as f:
                        index_data = json.load(f)
                        available.append(index_data)
        
        return sorted(available, key=lambda x: x.get('analysis_timestamp', ''), reverse=True)