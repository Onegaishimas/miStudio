                # Tokenize input
                inputs = tokenizer(
                    text, 
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(device)
                
                # Calculate loss
                outputs = model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss.item()
                
                # Calculate perplexity
                perplexity = torch.exp(outputs.loss).item()
                
                total_loss += loss
                total_perplexity += perplexity
                count += 1
                
            except Exception as e:
                print(f"Error processing text: {e}")
                continue
    
    # Return average loss (lower is better)
    average_loss = total_loss / count if count > 0 else float('inf')
    average_perplexity = total_perplexity / count if count > 0 else float('inf')
    
    # Combine loss and perplexity for comprehensive score
    return average_loss + (average_perplexity / 100.0)
EOF

echo "✅ Production benchmark function created"

# 4. Add ablation scoring to configuration
echo -e "\n4. ⚙️ Adding ablation scoring configuration..."
cat >> config/production_scoring.yaml << 'EOF'
  
  - scorer: "ablation_scorer"
    name: "model_utility"
    params:
      benchmark_dataset_path: "config/production_benchmark.py"
      target_model_name: "microsoft/phi-4"
      target_model_layer: "model.layers.16"
      device: "cuda"
EOF

echo "✅ Complete scoring configuration ready"

# 5. Execute comprehensive scoring
echo -e "\n5. 🚀 Executing comprehensive feature scoring..."
SCORING_REQUEST=$(cat << 'EOF'
{
  "features_path": "data/input/features.json",
  "config_path": "config/production_scoring.yaml", 
  "output_dir": "data/output"
}
EOF
)

SCORING_RESPONSE=$(curl -s -X POST "$BASE_URL/score" \
    -H "Content-Type: application/json" \
    -d "$SCORING_REQUEST")

echo "$SCORING_RESPONSE" | jq '.'

# Check if scoring was successful
SCORING_SUCCESS=$(echo "$SCORING_RESPONSE" | jq -r '.message' | grep -i "success")
OUTPUT_PATH=$(echo "$SCORING_RESPONSE" | jq -r '.output_path')

if [ -z "$SCORING_SUCCESS" ] || [ "$OUTPUT_PATH" == "null" ]; then
    echo "❌ Scoring failed"
    echo "$SCORING_RESPONSE" | jq '.detail'
    exit 1
fi

echo "✅ Scoring completed successfully"
echo "📁 Output file: $OUTPUT_PATH"

# 6. Analyze scoring results
echo -e "\n6. 📊 Analyzing scoring results..."

# Extract key metrics
FEATURES_SCORED=$(echo "$SCORING_RESPONSE" | jq -r '.features_scored')
SCORES_ADDED=$(echo "$SCORING_RESPONSE" | jq -r '.scores_added | join(", ")')

echo "Features processed: $FEATURES_SCORED"
echo "Scores generated: $SCORES_ADDED"

# Load and analyze results
if [ -f "$OUTPUT_PATH" ]; then
    echo -e "\n📈 Score Distribution Analysis:"
    
    # Safety score analysis
    echo "🛡️ AI Safety Scores:"
    jq '[.[] | select(.ai_safety_score != null) | .ai_safety_score] | sort | reverse' "$OUTPUT_PATH" | \
        jq -r 'to_entries | .[:5] | .[] | "  Top \(.key + 1): \(.value)"'
    
    # Technical complexity analysis  
    echo -e "\n🔧 Technical Complexity Scores:"
    jq '[.[] | select(.technical_complexity != null) | .technical_complexity] | sort | reverse' "$OUTPUT_PATH" | \
        jq -r 'to_entries | .[:5] | .[] | "  Top \(.key + 1): \(.value)"'
    
    # Business value analysis
    echo -e "\n💼 Business Value Scores:"
    jq '[.[] | select(.business_value != null) | .business_value] | sort | reverse' "$OUTPUT_PATH" | \
        jq -r 'to_entries | .[:5] | .[] | "  Top \(.key + 1): \(.value)"'
    
    # Model utility analysis
    echo -e "\n⚡ Model Utility Scores (Ablation):"
    jq '[.[] | select(.model_utility != null) | {index: .feature_index, utility: .model_utility}] | sort_by(.utility) | reverse' "$OUTPUT_PATH" | \
        jq -r '.[:5] | .[] | "  Feature \(.index): \(.utility)"'
    
    # Combined priority ranking
    echo -e "\n🎯 Top Priority Features (Combined Score):"
    jq '[.[] | {
        feature_index: .feature_index,
        safety: (.ai_safety_score // 0),
        technical: (.technical_complexity // 0), 
        business: (.business_value // 0),
        utility: ((.model_utility // 0) | if . < 0 then -. else . end),
        combined: (
            (.ai_safety_score // 0) * 0.3 +
            (.technical_complexity // 0) * 0.2 + 
            (.business_value // 0) * 0.3 +
            (((.model_utility // 0) | if . < 0 then -. else . end) * 0.2)
        )
    }] | sort_by(.combined) | reverse' "$OUTPUT_PATH" | \
        jq -r '.[:10] | .[] | "  Feature \(.feature_index): combined=\(.combined | floor * 1000 / 1000) (safety=\(.safety), tech=\(.technical), business=\(.business), utility=\(.utility))"'
    
else
    echo "❌ Output file not found: $OUTPUT_PATH"
fi

# 7. Generate summary report
echo -e "\n7. 📋 Generating summary report..."
SUMMARY_FILE="data/output/scoring_summary_$(date +%Y%m%d_%H%M%S).txt"

cat > "$SUMMARY_FILE" << EOF
miStudioScore Analysis Summary
=============================
Generated: $(date)
Features Analyzed: $FEATURES_SCORED
Scores Generated: $SCORES_ADDED
Output File: $OUTPUT_PATH

Score Categories:
- AI Safety Score: Measures alignment with safety and ethical principles
- Technical Complexity: Assesses algorithmic and architectural sophistication  
- Business Value: Evaluates commercial and market relevance
- Model Utility: Quantifies causal impact on model performance

Analysis Complete. See detailed results in: $OUTPUT_PATH
EOF

echo "✅ Summary report saved: $SUMMARY_FILE"

echo -e "\n🎉 miStudioScore Analysis Complete!"
echo "📊 Key Deliverables:"
echo "  - Scored features: $OUTPUT_PATH"
echo "  - Summary report: $SUMMARY_FILE"
echo "  - Configuration: config/production_scoring.yaml"
echo "  - Benchmark: config/production_benchmark.py"
```

#### Batch Scoring Multiple Feature Sets

```bash
#!/bin/bash

# Batch process multiple feature sets with different scoring strategies
FEATURE_SETS=("security_features.json" "performance_features.json" "user_experience_features.json")
BASE_URL="http://localhost:8004"

echo "🔄 Batch Feature Scoring Workflow"
echo "================================="

declare -A SCORING_CONFIGS
declare -A JOB_RESULTS

# Define scoring strategies for different feature types
create_security_config() {
    cat > config/security_scoring.yaml << 'EOF'
scoring_jobs:
  - scorer: "relevance_scorer"
    name: "security_risk"
    params:
      positive_keywords: ["vulnerability", "exploit", "attack", "breach", "unauthorized"]
      negative_keywords: ["secure", "protected", "encrypted", "safe"]
  
  - scorer: "relevance_scorer"
    name: "privacy_impact"
    params:
      positive_keywords: ["personal", "private", "sensitive", "confidential", "pii"]
      negative_keywords: ["public", "anonymous", "aggregated"]
EOF
}

create_performance_config() {
    cat > config/performance_scoring.yaml << 'EOF'
scoring_jobs:
  - scorer: "relevance_scorer"
    name: "performance_impact"
    params:
      positive_keywords: ["speed", "efficient", "optimized", "fast", "performance"]
      negative_keywords: ["slow", "inefficient", "bottleneck", "latency"]
  
  - scorer: "ablation_scorer"
    name: "computational_utility"
    params:
      benchmark_dataset_path: "config/performance_benchmark.py"
      target_model_name: "microsoft/phi-4"
      target_model_layer: "model.layers.16"
      device: "cuda"
EOF
}

create_ux_config() {
    cat > config/ux_scoring.yaml << 'EOF'
scoring_jobs:
  - scorer: "relevance_scorer"
    name: "user_satisfaction"
    params:
      positive_keywords: ["user", "customer", "experience", "satisfaction", "intuitive"]
      negative_keywords: ["complex", "confusing", "difficult", "frustrating"]
  
  - scorer: "relevance_scorer"
    name: "accessibility"
    params:
      positive_keywords: ["accessible", "inclusive", "universal", "usable", "clear"]
      negative_keywords: ["exclusive", "limited", "restricted"]
EOF
}

# Create performance benchmark
cat > config/performance_benchmark.py << 'EOF'
import torch
import time

def run_benchmark(model, tokenizer, device):
    """Performance-focused benchmark measuring inference speed."""
    test_texts = [
        "Quick performance test for inference speed measurement.",
        "Another test sentence for latency evaluation.",
        "Final test case for throughput assessment."
    ]
    
    start_time = time.time()
    total_tokens = 0
    
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs)
            total_tokens += inputs.input_ids.shape[1]
    
    end_time = time.time()
    processing_time = end_time - start_time
    tokens_per_second = total_tokens / processing_time
    
    # Return inverse of tokens/second (lower is better for ablation)
    return 1.0 / tokens_per_second if tokens_per_second > 0 else float('inf')
EOF

# Create configurations
echo "📝 Creating specialized scoring configurations..."
create_security_config
create_performance_config  
create_ux_config

SCORING_CONFIGS["security_features.json"]="config/security_scoring.yaml"
SCORING_CONFIGS["performance_features.json"]="config/performance_scoring.yaml"
SCORING_CONFIGS["user_experience_features.json"]="config/ux_scoring.yaml"

# Process each feature set
for FEATURE_SET in "${FEATURE_SETS[@]}"; do
    echo -e "\n🚀 Processing $FEATURE_SET..."
    
    CONFIG_FILE=${SCORING_CONFIGS[$FEATURE_SET]}
    
    if [ ! -f "data/input/$FEATURE_SET" ]; then
        echo "⚠️ Feature set not found: data/input/$FEATURE_SET, skipping..."
        continue
    fi
    
    SCORING_REQUEST=$(cat << EOF
{
  "features_path": "data/input/$FEATURE_SET",
  "config_path": "$CONFIG_FILE",
  "output_dir": "data/output"
}
EOF
)
    
    RESPONSE=$(curl -s -X POST "$BASE_URL/score" \
        -H "Content-Type: application/json" \
        -d "$SCORING_REQUEST")
    
    SUCCESS=$(echo "$RESPONSE" | jq -r '.message' | grep -i "success")
    
    if [ -n "$SUCCESS" ]; then
        OUTPUT_PATH=$(echo "$RESPONSE" | jq -r '.output_path')
        FEATURES_COUNT=$(echo "$RESPONSE" | jq -r '.features_scored')
        
        JOB_RESULTS[$FEATURE_SET]="$OUTPUT_PATH"
        
        echo "✅ $FEATURE_SET: $FEATURES_COUNT features scored"
        echo "   Output: $OUTPUT_PATH"
    else
        echo "❌ $FEATURE_SET: Scoring failed"
        echo "$RESPONSE" | jq '.detail'
    fi
done

# Generate comparative analysis
echo -e "\n📊 Comparative Analysis Across Feature Sets:"

for FEATURE_SET in "${FEATURE_SETS[@]}"; do
    OUTPUT_PATH=${JOB_RESULTS[$FEATURE_SET]}
    
    if [ -n "$OUTPUT_PATH" ] && [ -f "$OUTPUT_PATH" ]; then
        echo -e "\n📈 $FEATURE_SET Results:"
        
        # Extract relevant scores based on feature set type
        case $FEATURE_SET in
            "security_features.json")
                echo "  🛡️ Security Risk Distribution:"
                jq '[.[] | select(.security_risk != null)] | group_by(.security_risk > 0.5) | [.[0] | length, .[1] | length]' "$OUTPUT_PATH" | \
                    jq -r '"    Low Risk: \(.[0]), High Risk: \(.[1])"'
                ;;
            "performance_features.json") 
                echo "  ⚡ Performance Impact Analysis:"
                jq '[.[] | select(.performance_impact != null) | .performance_impact] | [min, max, (add / length)]' "$OUTPUT_PATH" | \
                    jq -r '"    Min: \(.[0]), Max: \(.[1]), Avg: \(.[2])"'
                ;;
            "user_experience_features.json")
                echo "  👥 User Experience Metrics:"
                jq '[.[] | select(.user_satisfaction != null) | .user_satisfaction] | [min, max, (add / length)]' "$OUTPUT_PATH" | \
                    jq -r '"    Min: \(.[0]), Max: \(.[1]), Avg: \(.[2])"'
                ;;
        esac
    fi
done

echo -e "\n🎯 Batch scoring complete! Check individual output files for detailed analysis."
```

## Best Practices

### 1. Configuration Design

Design effective scoring configurations for different use cases:

```python
def create_domain_specific_configs():
    """Create scoring configurations for different domains."""
    
    # AI Safety Configuration
    safety_config = {
        "scoring_jobs": [
            {
                "scorer": "relevance_scorer",
                "name": "bias_detection",
                "params": {
                    "positive_keywords": ["bias", "unfair", "discriminatory", "prejudice"],
                    "negative_keywords": ["fair", "unbiased", "neutral", "balanced"]
                }
            },
            {
                "scorer": "relevance_scorer", 
                "name": "harmful_content",
                "params": {
                    "positive_keywords": ["harmful", "toxic", "dangerous", "malicious"],
                    "negative_keywords": ["safe", "beneficial", "helpful", "positive"]
                }
            }
        ]
    }
    
    # Technical Quality Configuration
    technical_config = {
        "scoring_jobs": [
            {
                "scorer": "relevance_scorer",
                "name": "code_quality",
                "params": {
                    "positive_keywords": ["efficient", "optimized", "clean", "readable"],
                    "negative_keywords": ["buggy", "inefficient", "messy", "legacy"]
                }
            },
            {
                "scorer": "ablation_scorer",
                "name": "functional_utility", 
                "params": {
                    "benchmark_dataset_path": "config/code_benchmark.py",
                    "target_model_name": "microsoft/phi-4",
                    "target_model_layer": "model.layers.20",
                    "device": "cuda"
                }
            }
        ]
    }
    
    # Business Impact Configuration
    business_config = {
        "scoring_jobs": [
            {
                "scorer": "relevance_scorer",
                "name": "revenue_relevance",
                "params": {
                    "positive_keywords": ["revenue", "profit", "sales", "monetization"],
                    "negative_keywords": ["cost", "expense", "overhead", "loss"]
                }
            },
            {
                "scorer": "relevance_scorer",
                "name": "customer_impact",
                "params": {
                    "positive_keywords": ["customer", "satisfaction", "retention", "experience"],
                    "negative_keywords": ["churn", "complaint", "dissatisfaction"]
                }
            }
        ]
    }
    
    return {
        "safety": safety_config,
        "technical": technical_config,
        "business": business_config
    }
```

### 2. Benchmark Function Design

Create effective benchmark functions for ablation scoring:

```python
def create_task_specific_benchmarks():
    """Examples of task-specific benchmark functions."""
    
    # Question Answering Benchmark
    qa_benchmark = '''
import torch
from datasets import load_dataset

def run_benchmark(model, tokenizer, device):
    """Question answering performance benchmark."""
    
    # Load a small subset of SQuAD for evaluation
    dataset = load_dataset("squad", split="validation[:50]")
    
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for item in dataset:
            context = item["context"][:500]  # Limit context length
            question = item["question"]
            answer = item["answers"]["text"][0]
            
            prompt = f"Context: {context}\\nQuestion: {question}\\nAnswer: {answer}"
            
            inputs = tokenizer(
                prompt, 
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(device)
            
            outputs = model(**inputs, labels=inputs.input_ids)
            total_loss += outputs.loss.item()
            count += 1
    
    return total_loss / count if count > 0 else float('inf')
'''
    
    # Code Generation Benchmark
    code_benchmark = '''
import torch

def run_benchmark(model, tokenizer, device):
    """Code generation performance benchmark."""
    
    code_prompts = [
        "def fibonacci(n):",
        "class Calculator:",
        "import numpy as np\\n\\ndef matrix_multiply(",
        "# Sort a list of numbers\\ndef sort_numbers(arr):",
        "try:\\n    result = divide(a, b)"
    ]
    
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for prompt in code_prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=128,
                truncation=True
            ).to(device)
            
            # Generate continuation
            outputs = model(**inputs, labels=inputs.input_ids)
            total_loss += outputs.loss.item()
            count += 1
    
    return total_loss / count if count > 0 else float('inf')
'''
    
    # Mathematical Reasoning Benchmark
    math_benchmark = '''
import torch

def run_benchmark(model, tokenizer, device):
    """Mathematical reasoning benchmark."""
    
    math_problems = [
        "What is 15 + 27? Answer: 42",
        "If x + 5 = 12, then x = 7",
        "The area of a circle with radius 3 is 9π",
        "Solve: 2x - 6 = 10. Answer: x = 8",
        "What is the derivative of x²? Answer: 2x"
    ]
    
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for problem in math_problems:
            inputs = tokenizer(
                problem,
                return_tensors="pt", 
                max_length=256,
                truncation=True
            ).to(device)
            
            outputs = model(**inputs, labels=inputs.input_ids)
            total_loss += outputs.loss.item()
            count += 1
    
    return total_loss / count if count > 0 else float('inf')
'''
    
    return {
        "qa_benchmark.py": qa_benchmark,
        "code_benchmark.py": code_benchmark,
        "math_benchmark.py": math_benchmark
    }
```

### 3. Score Interpretation

Understand and interpret scoring results effectively:

```python
def analyze_scoring_results(scored_features):
    """Comprehensive analysis of scoring results."""
    
    # Statistical analysis
    scores_analysis = {}
    
    for feature in scored_features:
        for key, value in feature.items():
            if key.endswith('_score') or key.endswith('_relevance') or key.endswith('_utility'):
                if key not in scores_analysis:
                    scores_analysis[key] = []
                if value is not None:
                    scores_analysis[key].append(value)
    
    # Calculate statistics for each score type
    statistics = {}
    for score_name, values in scores_analysis.items():
        if values:
            statistics[score_name] = {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
            }
    
    # Feature ranking by combined criteria
    ranked_features = []
    for feature in scored_features:
        feature_scores = {k: v for k, v in feature.items() 
                         if (k.endswith('_score') or k.endswith('_relevance') or k.endswith('_utility')) 
                         and v is not None}
        
        if feature_scores:
            # Normalize scores to 0-1 range for combination
            normalized_scores = {}
            for score_name, value in feature_scores.items():
                if score_name in statistics:
                    stat = statistics[score_name]
                    if stat['max'] != stat['min']:
                        normalized = (value - stat['min']) / (stat['max'] - stat['min'])
                    else:
                        normalized = 0.5
                    normalized_scores[score_name] = normalized
            
            # Calculate combined priority score
            combined_score = sum(normalized_scores.values()) / len(normalized_scores)
            
            ranked_features.append({
                'feature_index': feature.get('feature_index'),
                'combined_score': combined_score,
                'individual_scores': feature_scores,
                'normalized_scores': normalized_scores
            })
    
    # Sort by combined score
    ranked_features.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return {
        'statistics': statistics,
        'ranked_features': ranked_features[:20],  # Top 20
        'recommendations': generate_recommendations(statistics, ranked_features)
    }

def generate_recommendations(statistics, ranked_features):
    """Generate actionable recommendations based on scoring results."""
    
    recommendations = []
    
    # High-priority features
    high_priority = [f for f in ranked_features if f['combined_score'] > 0.8]
    if high_priority:
        recommendations.append({
            'type': 'high_priority',
            'count': len(high_priority),
            'action': 'Focus immediate attention on these features for safety/performance analysis'
        })
    
    # Score distribution analysis
    for score_name, stats in statistics.items():
        if 'safety' in score_name.lower() or 'risk' in score_name.lower():
            high_risk_count = sum(1 for f in ranked_features 
                                 if f['individual_scores'].get(score_name, 0) > 0.5)
            if high_risk_count > 0:
                recommendations.append({
                    'type': 'safety_concern',
                    'score': score_name,
                    'count': high_risk_count,
                    'action': 'Review features with high safety/risk scores for potential issues'
                })
        
        elif 'utility' in score_name.lower():
            # For utility scores, large absolute values (positive or negative) are interesting
            high_utility = sum(1 for f in ranked_features 
                              if abs(f['individual_scores'].get(score_name, 0)) > 0.1)
            if high_utility > 0:
                recommendations.append({
                    'type': 'utility_analysis',
                    'score': score_name,
                    'count': high_utility,
                    'action': 'Investigate features with high utility impact on model performance'
                })
    
    return recommendations
```

### 4. Error Handling

Implement robust error handling for production use:

```python
def robust_scoring_workflow(client, features_path, config_path, output_dir, max_retries=3):
    """Execute scoring with comprehensive error handling."""
    
    for attempt in range(max_retries):
        try:
            # Validate inputs
            if not os.path.exists(features_path):
                raise FileNotFoundError(f"Features file not found: {features_path}")
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Execute scoring
            result = client.score_features(features_path, config_path, output_dir)
            
            # Validate output
            if not os.path.exists(result['output_path']):
                raise RuntimeError("Output file was not created")
            
            # Verify output file is valid JSON
            with open(result['output_path'], 'r') as f:
                scored_features = json.load(f)
            
            if not isinstance(scored_features, list):
                raise ValueError("Output file does not contain a list of features")
            
            return result
            
        except requests.exceptions.RequestException as e:
            if "CUDA out of memory" in str(e):
                print(f"GPU memory error on attempt {attempt + 1}, reducing batch size...")
                # Could implement logic to split processing into smaller batches
                continue
            elif "timeout" in str(e).lower():
                print(f"Timeout on attempt {attempt + 1}, retrying...")
                time.sleep(10 * (attempt + 1))  # Exponential backoff
                continue
            else:
                raise
        
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            # These are typically not recoverable
            raise
        
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Scoring failed after {max_retries} attempts: {e}")
            
            print(f"Attempt {attempt + 1} failed: {e}, retrying...")
            time.sleep(5 * (attempt + 1))

def validate_scoring_config(config_path):
    """Validate scoring configuration before execution."""
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Invalid YAML configuration: {e}")
    
    if 'scoring_jobs' not in config:
        raise ValueError("Configuration missing 'scoring_jobs' section")
    
    required_fields = ['scorer', 'name', 'params']
    
    for i, job in enumerate(config['scoring_jobs']):
        for field in required_fields:
            if field not in job:
                raise ValueError(f"Job {i} missing required field: {field}")
        
        # Validate scorer-specific parameters
        if job['scorer'] == 'relevance_scorer':
            params = job['params']
            if 'positive_keywords' not in params:
                raise ValueError(f"Relevance scorer job {i} missing 'positive_keywords'")
        
        elif job['scorer'] == 'ablation_scorer':
            params = job['params']
            required_ablation_params = ['benchmark_dataset_path', 'target_model_name', 'target_model_layer']
            for param in required_ablation_params:
                if param not in params:
                    raise ValueError(f"Ablation scorer job {i} missing '{param}'")
            
            # Validate benchmark file exists
            if not os.path.exists(params['benchmark_dataset_path']):
                raise FileNotFoundError(f"Benchmark file not found: {params['benchmark_dataset_path']}")
    
    return True
```

## Integration with miStudio Pipeline

### Upstream Integration (miStudioFind)

```python
def seamless_find_to_score_workflow(find_client, score_client, find_job_id):
    """Complete workflow from feature analysis to scoring."""
    
    # 1. Get miStudioFind results
    find_results = find_client.get_job_results(find_job_id)
    
    if not find_results.get('ready_for_explain_service'):
        raise Exception("Find results not ready for downstream processing")
    
    print(f"🔍 Find analysis complete: {find_results['summary']['interpretable_features']} interpretable features")
    
    # 2. Save features in scoring format
    features_path = f"data/input/features_{find_job_id}.json"
    with open(features_path, 'w') as f:
        json.dump(find_results['results'], f, indent=2)
    
    # 3. Create adaptive scoring configuration based on feature characteristics
    feature_categories = {}
    for feature in find_results['results']:
        category = feature.get('pattern_category', 'unknown')
        feature_categories[category] = feature_categories.get(category, 0) + 1
    
    # Adaptive configuration based on feature composition
    scoring_jobs = []
    
    # Always include business relevance
    scoring_jobs.append({
        "scorer": "relevance_scorer",
        "name": "business_relevance",
        "params": {
            "positive_keywords": ["important", "critical", "valuable", "significant"],
            "negative_keywords": ["trivial", "insignificant", "irrelevant"]
        }
    })
    
    # Add category-specific scoring
    if feature_categories.get('technical', 0) > 5:
        scoring_jobs.append({
            "scorer": "relevance_scorer",
            "name": "technical_sophistication",
            "params": {
                "positive_keywords": ["algorithm", "optimization", "architecture", "performance"],
                "negative_keywords": ["simple", "basic", "trivial"]
            }
        })
    
    if feature_categories.get('behavioral', 0) > 3:
        scoring_jobs.append({
            "scorer": "relevance_scorer", 
            "name": "behavioral_significance",
            "params": {
                "positive_keywords": ["decision", "behavior", "reasoning", "capability"],
                "negative_keywords": ["random", "noise", "artifact"]
            }
        })
    
    # Add ablation scoring for high-quality features
    interpretable_count = find_results['summary']['interpretable_features']
    if interpretable_count >= 10:
        scoring_jobs.append({
            "scorer": "ablation_scorer",
            "name": "causal_utility",
            "params": {
                "benchmark_dataset_path": "config/adaptive_benchmark.py",
    # Add ablation scoring for high-quality features
    interpretable_count = find_results['summary']['interpretable_features']
    if interpretable_count >= 10:
        scoring_jobs.append({
            "scorer": "ablation_scorer",
            "name": "causal_utility",
            "params": {
                "benchmark_dataset_path": "config/adaptive_benchmark.py",
                "target_model_name": find_results.get('model_name', 'microsoft/phi-4'),
                "target_model_layer": "model.layers.16",
                "device": "cuda"
            }
        })
    
    # Save adaptive configuration
    adaptive_config = {"scoring_jobs": scoring_jobs}
    config_path = f"config/adaptive_scoring_{find_job_id}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(adaptive_config, f, indent=2)
    
    # 4. Execute scoring
    score_result = score_client.score_features(
        features_path=features_path,
        config_path=config_path,
        output_dir="data/output"
    )
    
    print(f"📊 Scoring complete: {score_result['features_scored']} features scored")
    
    return {
        'find_job_id': find_job_id,
        'score_output_path': score_result['output_path'],
        'features_analyzed': find_results['summary']['total_features_analyzed'],
        'interpretable_features': interpretable_count,
        'scores_added': score_result['scores_added'],
        'ready_for_downstream': True
    }
```

### Downstream Integration (miStudioCorrelate)

```python
def prepare_for_correlation_service(score_results):
    """Prepare miStudioScore results for correlation analysis."""
    
    with open(score_results['score_output_path'], 'r') as f:
        scored_features = json.load(f)
    
    # Filter for high-scoring features across different dimensions
    correlation_candidates = []
    
    for feature in scored_features:
        # Extract all numerical scores
        scores = {}
        for key, value in feature.items():
            if (key.endswith('_score') or key.endswith('_relevance') or 
                key.endswith('_utility') or key.endswith('_significance')) and \
               isinstance(value, (int, float)):
                scores[key] = value
        
        # Calculate overall importance
        if scores:
            max_score = max(abs(v) for v in scores.values())
            mean_score = sum(abs(v) for v in scores.values()) / len(scores)
            
            # Include features with high importance or interesting patterns
            if max_score > 0.3 or mean_score > 0.2:
                correlation_candidates.append({
                    'feature_index': feature['feature_index'],
                    'scores': scores,
                    'max_score': max_score,
                    'mean_score': mean_score,
                    'pattern_category': feature.get('pattern_category', 'unknown'),
                    'coherence_score': feature.get('coherence_score', 0)
                })
    
    # Sort by importance
    correlation_candidates.sort(key=lambda x: x['max_score'], reverse=True)
    
    return {
        'source_score_job': score_results['score_output_path'],
        'correlation_candidates': correlation_candidates[:50],  # Top 50 for correlation
        'score_dimensions': list(set(
            key for candidate in correlation_candidates 
            for key in candidate['scores'].keys()
        )),
        'ready_for_correlation': len(correlation_candidates) >= 10
    }
```

## Advanced Features

### Custom Scorer Development

Create your own scorer modules:

```python
# src/scorers/custom_scorer.py
from typing import List, Dict, Any
from .base_scorer import BaseScorer
import numpy as np

class CustomBusinessScorer(BaseScorer):
    """Custom scorer for business-specific feature evaluation."""
    
    @property
    def name(self) -> str:
        return "custom_business_scorer"
    
    def score(self, features: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Calculate custom business scores based on multiple criteria.
        
        Args:
            features: List of feature dictionaries
            **kwargs: Configuration parameters
        
        Returns:
            Features with added custom scores
        """
        score_name = kwargs.get("name", "business_score")
        
        # Extract scoring parameters
        revenue_keywords = set(kwargs.get("revenue_keywords", []))
        cost_keywords = set(kwargs.get("cost_keywords", []))
        customer_keywords = set(kwargs.get("customer_keywords", []))
        risk_keywords = set(kwargs.get("risk_keywords", []))
        
        # Scoring weights
        weights = kwargs.get("weights", {
            "revenue_impact": 0.4,
            "cost_efficiency": 0.2,
            "customer_value": 0.3,
            "risk_factor": 0.1
        })
        
        for feature in features:
            # Get activation text
            activating_texts = feature.get("top_activating_examples", [])
            if not activating_texts:
                feature[score_name] = 0.0
                continue
            
            combined_text = " ".join(activating_texts).lower()
            
            # Calculate component scores
            revenue_score = self._calculate_keyword_score(combined_text, revenue_keywords)
            cost_score = self._calculate_keyword_score(combined_text, cost_keywords) 
            customer_score = self._calculate_keyword_score(combined_text, customer_keywords)
            risk_score = self._calculate_keyword_score(combined_text, risk_keywords)
            
            # Combine scores with weights
            business_score = (
                revenue_score * weights["revenue_impact"] +
                (1 - cost_score) * weights["cost_efficiency"] +  # Lower cost = better
                customer_score * weights["customer_value"] +
                risk_score * weights["risk_factor"]
            )
            
            feature[score_name] = round(business_score, 4)
            
            # Add component scores for transparency
            feature[f"{score_name}_components"] = {
                "revenue_impact": round(revenue_score, 3),
                "cost_efficiency": round(1 - cost_score, 3),
                "customer_value": round(customer_score, 3),
                "risk_factor": round(risk_score, 3)
            }
        
        return features
    
    def _calculate_keyword_score(self, text: str, keywords: set) -> float:
        """Calculate normalized keyword presence score."""
        if not keywords:
            return 0.0
        
        matches = sum(1 for keyword in keywords if keyword in text)
        return matches / len(keywords)

# Configuration example for custom scorer
custom_config = {
    "scoring_jobs": [
        {
            "scorer": "custom_business_scorer",
            "name": "comprehensive_business_value",
            "params": {
                "revenue_keywords": ["revenue", "sales", "profit", "monetization", "pricing"],
                "cost_keywords": ["cost", "expense", "overhead", "maintenance", "resource"],
                "customer_keywords": ["customer", "user", "satisfaction", "experience", "retention"],
                "risk_keywords": ["risk", "vulnerability", "compliance", "regulation", "audit"],
                "weights": {
                    "revenue_impact": 0.35,
                    "cost_efficiency": 0.25,
                    "customer_value": 0.30,
                    "risk_factor": 0.10
                }
            }
        }
    ]
}
```

### Multi-Model Ablation Scoring

Advanced ablation scoring across multiple models:

```python
# config/multi_model_benchmark.py
import torch
from transformers import AutoModel, AutoTokenizer

def run_benchmark(model, tokenizer, device):
    """Multi-model comparative benchmark."""
    
    # Load comparison models
    comparison_models = [
        "microsoft/DialoGPT-small",
        "microsoft/phi-2", 
        "microsoft/phi-4"
    ]
    
    test_cases = [
        "Explain the concept of machine learning.",
        "What are the benefits of artificial intelligence?",
        "How does natural language processing work?"
    ]
    
    # Calculate performance on main model
    main_performance = calculate_model_performance(model, tokenizer, test_cases, device)
    
    # Calculate average performance across comparison models
    comparison_performances = []
    for model_name in comparison_models:
        try:
            comp_model = AutoModel.from_pretrained(model_name).to(device)
            comp_tokenizer = AutoTokenizer.from_pretrained(model_name)
            comp_performance = calculate_model_performance(comp_model, comp_tokenizer, test_cases, device)
            comparison_performances.append(comp_performance)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
    
    # Return relative performance (how much better/worse than average)
    if comparison_performances:
        avg_comparison = sum(comparison_performances) / len(comparison_performances)
        return main_performance - avg_comparison
    else:
        return main_performance

def calculate_model_performance(model, tokenizer, test_cases, device):
    """Calculate performance for a single model."""
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for text in test_cases:
            try:
                inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).to(device)
                outputs = model(**inputs, labels=inputs.input_ids)
                total_loss += outputs.loss.item()
                count += 1
            except Exception:
                continue
    
    return total_loss / count if count > 0 else float('inf')
```

### Scoring Analytics and Visualization

Generate insights from scoring results:

```python
def generate_scoring_analytics(scored_features_path):
    """Generate comprehensive analytics from scoring results."""
    
    with open(scored_features_path, 'r') as f:
        features = json.load(f)
    
    # Extract all score types
    score_types = set()
    for feature in features:
        for key in feature.keys():
            if (key.endswith('_score') or key.endswith('_relevance') or 
                key.endswith('_utility') or key.endswith('_significance')):
                score_types.add(key)
    
    analytics = {
        'overview': {
            'total_features': len(features),
            'score_types': list(score_types),
            'features_with_scores': sum(1 for f in features if any(k in score_types for k in f.keys()))
        },
        'score_distributions': {},
        'correlations': {},
        'feature_rankings': {},
        'category_analysis': {}
    }
    
    # Score distributions
    for score_type in score_types:
        values = [f[score_type] for f in features if score_type in f and f[score_type] is not None]
        if values:
            analytics['score_distributions'][score_type] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'percentiles': {
                    '25': np.percentile(values, 25),
                    '50': np.percentile(values, 50),
                    '75': np.percentile(values, 75),
                    '95': np.percentile(values, 95)
                }
            }
    
    # Score correlations
    score_pairs = [(s1, s2) for i, s1 in enumerate(score_types) for s2 in list(score_types)[i+1:]]
    for s1, s2 in score_pairs:
        values1 = []
        values2 = []
        for feature in features:
            if s1 in feature and s2 in feature and feature[s1] is not None and feature[s2] is not None:
                values1.append(feature[s1])
                values2.append(feature[s2])
        
        if len(values1) > 10:  # Need sufficient data points
            correlation = np.corrcoef(values1, values2)[0, 1]
            if not np.isnan(correlation):
                analytics['correlations'][f"{s1}_vs_{s2}"] = {
                    'correlation': correlation,
                    'sample_size': len(values1),
                    'strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.4 else 'weak'
                }
    
    # Feature rankings by each score
    for score_type in score_types:
        ranked = sorted(
            [f for f in features if score_type in f and f[score_type] is not None],
            key=lambda x: abs(x[score_type]) if 'utility' in score_type else x[score_type],
            reverse=True
        )
        analytics['feature_rankings'][score_type] = [
            {
                'feature_index': f['feature_index'],
                'score': f[score_type],
                'category': f.get('pattern_category', 'unknown')
            }
            for f in ranked[:10]
        ]
    
    # Category analysis
    categories = {}
    for feature in features:
        category = feature.get('pattern_category', 'unknown')
        if category not in categories:
            categories[category] = []
        categories[category].append(feature)
    
    for category, cat_features in categories.items():
        category_scores = {}
        for score_type in score_types:
            values = [f[score_type] for f in cat_features if score_type in f and f[score_type] is not None]
            if values:
                category_scores[score_type] = {
                    'mean': np.mean(values),
                    'count': len(values)
                }
        
        analytics['category_analysis'][category] = {
            'feature_count': len(cat_features),
            'average_scores': category_scores
        }
    
    return analytics

def generate_scoring_report(analytics, output_path):
    """Generate a comprehensive scoring report."""
    
    report = f"""
# miStudioScore Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Total Features Analyzed**: {analytics['overview']['total_features']}
- **Score Types Generated**: {len(analytics['overview']['score_types'])}
- **Features with Scores**: {analytics['overview']['features_with_scores']}

## Score Distributions

"""
    
    for score_type, dist in analytics['score_distributions'].items():
        report += f"""
### {score_type.replace('_', ' ').title()}

- Count: {dist['count']} features
- Mean: {dist['mean']:.3f}
- Range: {dist['min']:.3f} to {dist['max']:.3f}
- 95th Percentile: {dist['percentiles']['95']:.3f}

"""
    
    report += "\n## Top Features by Score\n"
    
    for score_type, rankings in analytics['feature_rankings'].items():
        report += f"\n### {score_type.replace('_', ' ').title()}\n\n"
        for i, feature in enumerate(rankings[:5], 1):
            report += f"{i}. Feature {feature['feature_index']}: {feature['score']:.3f} ({feature['category']})\n"
    
    if analytics['correlations']:
        report += "\n## Score Correlations\n\n"
        strong_correlations = {k: v for k, v in analytics['correlations'].items() if v['strength'] == 'strong'}
        if strong_correlations:
            report += "### Strong Correlations (|r| > 0.7)\n\n"
            for pair, corr in strong_correlations.items():
                report += f"- {pair.replace('_vs_', ' vs ')}: r = {corr['correlation']:.3f}\n"
    
    report += "\n## Category Analysis\n\n"
    
    for category, analysis in analytics['category_analysis'].items():
        report += f"### {category.title()} Features ({analysis['feature_count']} features)\n\n"
        for score_type, score_data in analysis['average_scores'].items():
            report += f"- Average {score_type}: {score_data['mean']:.3f}\n"
        report += "\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    return output_path
```

## Troubleshooting

### Common Issues

#### 1. Configuration Errors
```
Error: Configuration validation failed: scorer 'invalid_scorer' not found
```

**Solutions**:
- Check available scorers: `GET /health`
- Verify scorer names in configuration
- Ensure scorer modules are properly loaded

#### 2. Benchmark Function Errors
```
Error: run_benchmark function not found in benchmark_dataset.py
```

**Solutions**:
```python
# Ensure benchmark file has correct structure
def run_benchmark(model, tokenizer, device):
    """Required function signature"""
    # Implementation here
    return performance_score
```

#### 3. Memory Issues During Ablation
```
Error: CUDA out of memory during ablation scoring
```

**Solutions**:
- Use smaller batch sizes in benchmark
- Process features individually
- Use CPU device for large models
- Clear GPU cache between features

### Debugging Tools

```python
def debug_scoring_workflow(features_path, config_path):
    """Debug scoring workflow step by step."""
    
    print("🔍 Debugging miStudioScore Workflow")
    
    # 1. Validate input files
    try:
        with open(features_path, 'r') as f:
            features = json.load(f)
        print(f"✅ Features file loaded: {len(features)} features")
    except Exception as e:
        print(f"❌ Features file error: {e}")
        return
    
    # 2. Validate configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config loaded: {len(config.get('scoring_jobs', []))} jobs")
        
        for i, job in enumerate(config.get('scoring_jobs', [])):
            print(f"   Job {i}: {job.get('scorer')} -> {job.get('name')}")
    except Exception as e:
        print(f"❌ Config file error: {e}")
        return
    
    # 3. Test individual scorers
    from src.scorers.relevance_scorer import RelevanceScorer
    from src.scorers.ablation_scorer import AblationScorer
    
    available_scorers = {
        'relevance_scorer': RelevanceScorer,
        'ablation_scorer': AblationScorer
    }
    
    for job in config.get('scoring_jobs', []):
        scorer_name = job.get('scorer')
        if scorer_name in available_scorers:
            print(f"✅ Scorer {scorer_name} available")
            
            # Test scorer instantiation
            try:
                scorer = available_scorers[scorer_name]()
                print(f"   Scorer name: {scorer.name}")
            except Exception as e:
                print(f"❌ Scorer instantiation failed: {e}")
        else:
            print(f"❌ Scorer {scorer_name} not found")
    
    # 4. Test sample scoring
    sample_features = features[:2]  # Test with first 2 features
    
    for job in config.get('scoring_jobs', []):
        scorer_name = job.get('scorer')
        if scorer_name == 'relevance_scorer':  # Test relevance scorer only
            try:
                scorer = RelevanceScorer()
                params = {'name': 'test_score', **job.get('params', {})}
                
                result = scorer.score(sample_features.copy(), **params)
                print(f"✅ {scorer_name} test successful")
                
                # Check if scores were added
                for feature in result:
                    if 'test_score' in feature:
                        print(f"   Sample score: {feature['test_score']}")
                        break
                        
            except Exception as e:
                print(f"❌ {scorer_name} test failed: {e}")
```

## Conclusion

This comprehensive API reference provides everything needed to integrate with the miStudioScore service. The service offers:

- **Quantitative Feature Assessment**: Transform qualitative features into ranked, scored data
- **Flexible Scoring Methods**: Keyword-based relevance and causal utility measurement
- **Configuration-Driven**: YAML-based scoring pipeline definition
- **Modular Architecture**: Extensible scorer framework for custom algorithms
- **Production Ready**: Robust error handling and comprehensive testing

### Key Benefits

1. **Data-Driven Prioritization**: Numerical scores enable objective feature ranking
2. **Business Alignment**: Relevance scoring connects features to business objectives
3. **Causal Understanding**: Ablation scoring measures actual impact on model performance
4. **Flexible Configuration**: Adapt scoring strategies to different use cases
5. **Pipeline Integration**: Seamless flow from miStudioFind to downstream services

### Next Steps

For production deployment:
1. Set up configuration management for different scoring strategies
2. Implement custom scorers for domain-specific requirements
3. Establish benchmark functions for ablation scoring
4. Create monitoring and alerting for scoring jobs
5. Integrate with downstream correlation and monitoring services

For technical support and advanced configuration, refer to the interactive API documentation at `/docs` when the service is running.

---

**Service Status**: ✅ Complete Implementation  
**Last Updated**: July 27, 2025  
**API Version**: 1.0.0# miStudioScore API Reference Documentation

## Overview

miStudioScore is the **quantitative feature scoring engine** in the miStudio AI Interpretability Platform. It moves beyond qualitative analysis to provide data-driven answers to the critical question: "How important is this feature?" The service enriches feature data from miStudioFind with numerical scores, enabling users to rank, filter, and prioritize features based on their causal utility, safety implications, or business relevance.

**Service Status**: ✅ **Complete Implementation**  
**Version**: 1.0.0  
**Base URL**: `http://<host>:8004`  
**Documentation**: `/docs` (OpenAPI/Swagger)

## Service Architecture

miStudioScore processes feature data through a configurable scoring pipeline:

```
miStudioFind Features → Configuration-Driven Scoring → Quantitative Feature Rankings
```

### Core Capabilities

- **Relevance Scoring**: Keyword-based business relevance assessment
- **Ablation Scoring**: Causal utility measurement through feature ablation
- **Configurable Pipeline**: YAML-driven scoring job orchestration
- **Modular Scorers**: Pluggable scoring algorithms for extensibility
- **Quantitative Ranking**: Numerical scores for feature prioritization

## Quick Start

### 1. Prepare Input Files

You need three input files:

```bash
# features.json (from miStudioFind)
data/input/features.json

# scoring configuration
config/scoring_config.yaml

# benchmark dataset (for ablation scoring)
config/benchmark_dataset.py
```

### 2. Create Scoring Configuration

```yaml
# config/scoring_config.yaml
scoring_jobs:
  - scorer: "relevance_scorer"
    name: "security_relevance"
    params:
      positive_keywords: ["security", "encryption", "authentication", "vulnerability"]
      negative_keywords: ["marketing", "social", "entertainment"]
  
  - scorer: "ablation_scorer"
    name: "qa_utility"
    params:
      benchmark_dataset_path: "config/benchmark_dataset.py"
      target_model_name: "microsoft/phi-4"
      target_model_layer: "model.layers.16"
      device: "cuda"
```

### 3. Execute Scoring

```bash
curl -X POST "http://localhost:8004/score" \
  -H "Content-Type: application/json" \
  -d '{
    "features_path": "data/input/features.json",
    "config_path": "config/scoring_config.yaml",
    "output_dir": "data/output"
  }'
```

### 4. Retrieve Results

The service generates a timestamped output file:
```bash
data/output/scores_20250727143022.json
```

## API Endpoints

### Core Service Endpoints

#### Service Information

```http
GET /
```

**Description**: Get service information and available scorers.

**Response Example**:
```json
{
  "service": "miStudioScore",
  "status": "running",
  "version": "1.0.0",
  "description": "A service for scoring features from a trained SAE",
  "available_scorers": ["relevance_scorer", "ablation_scorer"],
  "configuration_driven": true,
  "documentation": "/docs"
}
```

#### Health Check

```http
GET /health
```

**Description**: Service health status and system diagnostics.

**Response Example**:
```json
{
  "status": "ok",
  "service": "miStudioScore",
  "version": "1.0.0",
  "timestamp": "2025-07-27T14:30:22.123456",
  "system_health": {
    "scorer_modules_loaded": 2,
    "configuration_valid": true,
    "dependencies_available": true
  },
  "available_scorers": {
    "relevance_scorer": "loaded",
    "ablation_scorer": "loaded"
  }
}
```

### Feature Scoring

#### Score Features

```http
POST /score
```

**Description**: Execute feature scoring based on configuration.

**Request Body**:
```json
{
  "features_path": "data/input/features.json",
  "config_path": "config/scoring_config.yaml",
  "output_dir": "data/output"
}
```

**Request Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `features_path` | string | Yes | Path to input features.json from miStudioFind |
| `config_path` | string | Yes | Path to scoring configuration YAML file |
| `output_dir` | string | Yes | Directory for output scores file |

**Response Example (200 OK)**:
```json
{
  "message": "Scoring job completed successfully.",
  "output_path": "data/output/scores_20250727143022.json",
  "features_scored": 512,
  "scores_added": ["security_relevance", "qa_utility"],
  "processing_summary": {
    "total_scoring_jobs": 2,
    "successful_jobs": 2,
    "failed_jobs": 0,
    "processing_time_seconds": 127.4
  }
}
```

**Error Responses**:
```json
// 400 Bad Request - Invalid file paths
{
  "detail": "Input file not found: data/input/missing_features.json"
}

// 400 Bad Request - Configuration error
{
  "detail": "Configuration validation failed: scorer 'invalid_scorer' not found"
}

// 500 Internal Server Error - Processing failure
{
  "detail": "Scoring job failed: CUDA out of memory during ablation scoring"
}
```

## Configuration Reference

### Scoring Configuration File

The `scoring_config.yaml` file defines the scoring pipeline:

```yaml
scoring_jobs:
  - scorer: "relevance_scorer"          # Scorer module name
    name: "business_relevance"          # Score field name in output
    params:                             # Scorer-specific parameters
      positive_keywords: ["important", "critical"]
      negative_keywords: ["irrelevant", "noise"]
  
  - scorer: "ablation_scorer"
    name: "causal_utility"
    params:
      benchmark_dataset_path: "config/qa_benchmark.py"
      target_model_name: "microsoft/phi-4"
      target_model_layer: "model.layers.16"
      device: "cuda"
```

### Configuration Schema

```typescript
interface ScoringConfig {
  scoring_jobs: ScoringJob[];
}

interface ScoringJob {
  scorer: string;                    // Scorer module identifier
  name: string;                      // Output score field name
  params: ScorerParams;              // Scorer-specific parameters
}

type ScorerParams = RelevanceScorerParams | AblationScorerParams;
```

## Scoring Methods

### 1. Relevance Scorer

**Purpose**: Keyword-based business relevance assessment

**Module**: `relevance_scorer`

**Parameters**:
```yaml
params:
  positive_keywords: ["security", "privacy", "safety"]    # Boost score
  negative_keywords: ["marketing", "spam", "irrelevant"]  # Reduce score
```

**Algorithm**:
1. Analyzes `top_activating_examples` for each feature
2. Counts positive and negative keyword occurrences
3. Calculates normalized score: `(positive_count - negative_count) / total_keywords`
4. Score range: -1.0 to 1.0

**Example Configuration**:
```yaml
- scorer: "relevance_scorer"
  name: "safety_relevance"
  params:
    positive_keywords: 
      - "safety"
      - "security" 
      - "privacy"
      - "ethical"
      - "responsible"
    negative_keywords:
      - "harmful"
      - "dangerous"
      - "malicious"
      - "toxic"
```

**Output Example**:
```json
{
  "feature_index": 348,
  "safety_relevance": 0.6,
  "top_activating_examples": ["Safety protocols...", "Security measures..."]
}
```

### 2. Ablation Scorer

**Purpose**: Measures causal utility through feature ablation

**Module**: `ablation_scorer`

**Parameters**:
```yaml
params:
  benchmark_dataset_path: "config/benchmark_dataset.py"  # Benchmark function
  target_model_name: "microsoft/phi-4"                   # Model to evaluate
  target_model_layer: "model.layers.16"                  # Layer to ablate
  device: "cuda"                                          # Compute device
```

**Algorithm**:
1. Loads target model and benchmark function
2. Calculates baseline performance score
3. For each feature:
   - Registers ablation hook on target layer
   - Zeroes out feature activation
   - Re-runs benchmark
   - Calculates utility = ablated_score - baseline_score
4. Removes hooks and returns utility scores

**Benchmark Function Requirements**:
```python
# config/benchmark_dataset.py
def run_benchmark(model, tokenizer, device):
    """
    Benchmark function for ablation scoring.
    
    Args:
        model: HuggingFace model instance
        tokenizer: Model tokenizer
        device: Compute device
    
    Returns:
        float: Performance score (lower is better for loss-based metrics)
    """
    # Implement your evaluation logic here
    # Examples: perplexity, accuracy, F1 score, etc.
    pass
```

**Example Benchmark Implementation**:
```python
# config/qa_benchmark.py
import torch
from datasets import load_dataset

def run_benchmark(model, tokenizer, device):
    """Question-answering benchmark."""
    
    # Load evaluation dataset
    dataset = load_dataset("squad", split="validation[:100]")
    
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for item in dataset:
            question = item["question"]
            context = item["context"]
            answer = item["answers"]["text"][0]
            
            # Prepare input
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
            
            # Calculate loss
            outputs = model(**inputs, labels=inputs.input_ids)
            total_loss += outputs.loss.item()
            count += 1
    
    return total_loss / count if count > 0 else float('inf')
```

**Output Example**:
```json
{
  "feature_index": 348,
  "qa_utility": -0.0234,
  "baseline_performance": 2.1456,
  "ablated_performance": 2.1690
}
```

## Data Models

### ScoreRequest

```typescript
interface ScoreRequest {
  features_path: string;             // Path to miStudioFind features.json
  config_path: string;               // Path to scoring configuration YAML
  output_dir: string;                // Output directory for scores file
}
```

### ScoreResponse

```typescript
interface ScoreResponse {
  message: string;                   // Success confirmation message
  output_path: string;               // Path to generated scores file
  features_scored: number;           // Number of features processed
  scores_added: string[];            // List of score names added
  processing_summary?: {
    total_scoring_jobs: number;
    successful_jobs: number;
    failed_jobs: number;
    processing_time_seconds: number;
  };
}
```

### Input Feature Format

Expected structure from miStudioFind:

```json
[
  {
    "feature_index": 348,
    "coherence_score": 0.501,
    "quality_level": "medium",
    "pattern_category": "technical",
    "pattern_keywords": ["json", "schema", "validation"],
    "top_activating_examples": [
      "JSON schema validation patterns for API documentation...",
      "Schema compliance checking in data processing..."
    ],
    "activation_statistics": {
      "mean": 0.15,
      "std": 0.08,
      "frequency": 0.023
    }
  }
]
```

### Output Scores Format

Features enriched with scoring results:

```json
[
  {
    "feature_index": 348,
    "coherence_score": 0.501,
    "pattern_category": "technical",
    "top_activating_examples": ["..."],
    "security_relevance": 0.4,         // Added by relevance scorer
    "qa_utility": -0.0156,             // Added by ablation scorer
    "business_priority": 0.8            // Added by custom scorer
  }
]
```

## Code Examples

### Python Client

```python
import requests
import json
import yaml
import time
from typing import Dict, Any, List

class MiStudioScoreClient:
    def __init__(self, base_url="http://localhost:8004"):
        self.base_url = base_url
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health and scorer availability."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def score_features(
        self,
        features_path: str,
        config_path: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """Execute feature scoring with configuration."""
        
        payload = {
            "features_path": features_path,
            "config_path": config_path,
            "output_dir": output_dir
        }
        
        response = requests.post(f"{self.base_url}/score", json=payload)
        response.raise_for_status()
        return response.json()
    
    def create_relevance_config(
        self,
        config_path: str,
        score_name: str,
        positive_keywords: List[str],
        negative_keywords: List[str] = None
    ):
        """Create a relevance scoring configuration."""
        
        config = {
            "scoring_jobs": [
                {
                    "scorer": "relevance_scorer",
                    "name": score_name,
                    "params": {
                        "positive_keywords": positive_keywords,
                        "negative_keywords": negative_keywords or []
                    }
                }
            ]
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, indent=2)
    
    def create_ablation_config(
        self,
        config_path: str,
        score_name: str,
        benchmark_path: str,
        model_name: str,
        layer_name: str,
        device: str = "cuda"
    ):
        """Create an ablation scoring configuration."""
        
        config = {
            "scoring_jobs": [
                {
                    "scorer": "ablation_scorer",
                    "name": score_name,
                    "params": {
                        "benchmark_dataset_path": benchmark_path,
                        "target_model_name": model_name,
                        "target_model_layer": layer_name,
                        "device": device
                    }
                }
            ]
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, indent=2)
    
    def create_benchmark_function(self, output_path: str, benchmark_type: str = "perplexity"):
        """Create a benchmark function for ablation scoring."""
        
        if benchmark_type == "perplexity":
            benchmark_code = '''
import torch

def run_benchmark(model, tokenizer, device):
    """Simple perplexity benchmark."""
    test_text = "The quick brown fox jumps over the lazy dog. This is a test sentence for evaluating model performance."
    
    inputs = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss
    
    return loss.item()
'''
        elif benchmark_type == "classification":
            benchmark_code = '''
import torch
from sklearn.metrics import accuracy_score

def run_benchmark(model, tokenizer, device):
    """Simple classification benchmark."""
    # Example classification prompts
    prompts = [
        ("This movie is amazing!", "positive"),
        ("This movie is terrible!", "negative"),
        ("The weather is nice today.", "neutral")
    ]
    
    correct_predictions = 0
    total_predictions = len(prompts)
    
    with torch.no_grad():
        for text, expected in prompts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs)
            
            # Simple prediction logic (customize based on your task)
            logits = outputs.logits
            predicted = torch.argmax(logits, dim=-1)
            
            # This is a simplified example - implement proper classification logic
            correct_predictions += 1 if predicted.item() > 0 else 0
    
    return correct_predictions / total_predictions
'''
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")
        
        with open(output_path, 'w') as f:
            f.write(benchmark_code)

# Usage example
def score_feature_importance():
    client = MiStudioScoreClient()
    
    # 1. Check service health
    health = client.health_check()
    print(f"Service status: {health['status']}")
    print(f"Available scorers: {health['available_scorers']}")
    
    # 2. Create benchmark function
    client.create_benchmark_function("config/perplexity_benchmark.py", "perplexity")
    
    # 3. Create comprehensive scoring configuration
    comprehensive_config = {
        "scoring_jobs": [
            {
                "scorer": "relevance_scorer",
                "name": "safety_relevance",
                "params": {
                    "positive_keywords": ["safety", "secure", "privacy", "ethical"],
                    "negative_keywords": ["harmful", "dangerous", "malicious"]
                }
            },
            {
                "scorer": "relevance_scorer", 
                "name": "technical_relevance",
                "params": {
                    "positive_keywords": ["function", "algorithm", "method", "system"],
                    "negative_keywords": ["marketing", "social", "entertainment"]
                }
            },
            {
                "scorer": "ablation_scorer",
                "name": "perplexity_utility",
                "params": {
                    "benchmark_dataset_path": "config/perplexity_benchmark.py",
                    "target_model_name": "microsoft/phi-4",
                    "target_model_layer": "model.layers.16",
                    "device": "cuda"
                }
            }
        ]
    }
    
    with open("config/comprehensive_scoring.yaml", 'w') as f:
        yaml.dump(comprehensive_config, f, indent=2)
    
    # 4. Execute scoring
    try:
        result = client.score_features(
            features_path="data/input/features.json",
            config_path="config/comprehensive_scoring.yaml",
            output_dir="data/output"
        )
        
        print(f"✅ Scoring completed successfully!")
        print(f"📁 Output file: {result['output_path']}")
        print(f"📊 Features scored: {result['features_scored']}")
        print(f"🏷️ Scores added: {', '.join(result['scores_added'])}")
        
        # 5. Load and analyze results
        with open(result['output_path'], 'r') as f:
            scored_features = json.load(f)
        
        # Find top features by different criteria
        safety_features = sorted(scored_features, 
                               key=lambda x: x.get('safety_relevance', 0), 
                               reverse=True)[:5]
        
        utility_features = sorted(scored_features,
                                key=lambda x: abs(x.get('perplexity_utility', 0)),
                                reverse=True)[:5]
        
        print(f"\n🛡️ Top 5 Safety-Relevant Features:")
        for i, feature in enumerate(safety_features):
            print(f"  {i+1}. Feature {feature['feature_index']}: "
                  f"safety={feature.get('safety_relevance', 0):.3f}")
        
        print(f"\n⚡ Top 5 High-Utility Features:")
        for i, feature in enumerate(utility_features):
            print(f"  {i+1}. Feature {feature['feature_index']}: "
                  f"utility={feature.get('perplexity_utility', 0):.3f}")
                  
    except requests.RequestException as e:
        print(f"❌ Scoring failed: {e}")

# Run the example
score_feature_importance()
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');
const fs = require('fs');
const yaml = require('js-yaml');

class MiStudioScoreClient {
    constructor(baseUrl = 'http://localhost:8004') {
        this.baseUrl = baseUrl;
        this.client = axios.create({ baseURL: baseUrl });
    }
    
    async healthCheck() {
        const response = await this.client.get('/health');
        return response.data;
    }
    
    async scoreFeatures(featuresPath, configPath, outputDir) {
        const payload = {
            features_path: featuresPath,
            config_path: configPath,
            output_dir: outputDir
        };
        
        const response = await this.client.post('/score', payload);
        return response.data;
    }
    
    createRelevanceConfig(outputPath, scoreName, positiveKeywords, negativeKeywords = []) {
        const config = {
            scoring_jobs: [
                {
                    scorer: 'relevance_scorer',
                    name: scoreName,
                    params: {
                        positive_keywords: positiveKeywords,
                        negative_keywords: negativeKeywords
                    }
                }
            ]
        };
        
        fs.writeFileSync(outputPath, yaml.dump(config, { indent: 2 }));
    }
    
    createBenchmarkFunction(outputPath, benchmarkType = 'perplexity') {
        let benchmarkCode;
        
        if (benchmarkType === 'perplexity') {
            benchmarkCode = `
import torch

def run_benchmark(model, tokenizer, device):
    """Perplexity benchmark for ablation scoring."""
    test_texts = [
        "The artificial intelligence system processes natural language with remarkable accuracy.",
        "Machine learning algorithms require substantial computational resources for training.",
        "Neural networks learn complex patterns from large datasets through iterative optimization."
    ]
    
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).to(device)
            outputs = model(**inputs, labels=inputs.input_ids)
            total_loss += outputs.loss.item()
            count += 1
    
    return total_loss / count if count > 0 else float('inf')
`;
        }
        
        fs.writeFileSync(outputPath, benchmarkCode);
    }
}

// Usage example
async function scoreBusinessRelevance() {
    const client = new MiStudioScoreClient();
    
    try {
        // 1. Health check
        const health = await client.healthCheck();
        console.log(`🟢 Service status: ${health.status}`);
        
        // 2. Create business-focused scoring configuration
        const businessConfig = {
            scoring_jobs: [
                {
                    scorer: 'relevance_scorer',
                    name: 'customer_focus',
                    params: {
                        positive_keywords: ['customer', 'user', 'service', 'support', 'satisfaction'],
                        negative_keywords: ['internal', 'backend', 'infrastructure']
                    }
                },
                {
                    scorer: 'relevance_scorer',
                    name: 'revenue_impact',
                    params: {
                        positive_keywords: ['revenue', 'sales', 'profit', 'monetization', 'pricing'],
                        negative_keywords: ['cost', 'expense', 'overhead']
                    }
                },
                {
                    scorer: 'relevance_scorer',
                    name: 'risk_assessment',
                    params: {
                        positive_keywords: ['risk', 'compliance', 'regulation', 'audit', 'security'],
                        negative_keywords: ['safe', 'secure', 'compliant']
                    }
                }
            ]
        };
        
        fs.writeFileSync('config/business_scoring.yaml', yaml.dump(businessConfig, { indent: 2 }));
        
        // 3. Execute business-focused scoring
        const result = await client.scoreFeatures(
            'data/input/features.json',
            'config/business_scoring.yaml',
            'data/output'
        );
        
        console.log('✅ Business scoring completed!');
        console.log(`📁 Results: ${result.output_path}`);
        console.log(`📊 Features analyzed: ${result.features_scored}`);
        console.log(`🏷️ Business metrics: ${result.scores_added.join(', ')}`);
        
        // 4. Analyze business relevance
        const scoredFeatures = JSON.parse(fs.readFileSync(result.output_path, 'utf8'));
        
        // Calculate business priority score
        const businessPriority = scoredFeatures.map(feature => ({
            feature_index: feature.feature_index,
            customer_focus: feature.customer_focus || 0,
            revenue_impact: feature.revenue_impact || 0,
            risk_assessment: Math.abs(feature.risk_assessment || 0), // Higher risk = higher priority
            business_score: (
                (feature.customer_focus || 0) * 0.4 +
                (feature.revenue_impact || 0) * 0.4 +
                Math.abs(feature.risk_assessment || 0) * 0.2
            )
        })).sort((a, b) => b.business_score - a.business_score);
        
        console.log('\n🎯 Top 10 Business-Critical Features:');
        businessPriority.slice(0, 10).forEach((feature, index) => {
            console.log(`  ${index + 1}. Feature ${feature.feature_index}: ` +
                       `score=${feature.business_score.toFixed(3)} ` +
                       `(customer=${feature.customer_focus.toFixed(2)}, ` +
                       `revenue=${feature.revenue_impact.toFixed(2)}, ` +
                       `risk=${feature.risk_assessment.toFixed(2)})`);
        });
        
    } catch (error) {
        console.error('❌ Business scoring failed:', error.message);
    }
}

// Run business relevance analysis
scoreBusinessRelevance();
```

### cURL Examples

#### Basic Scoring Workflow

```bash
#!/bin/bash

# miStudioScore Complete Workflow
BASE_URL="http://localhost:8004"

echo "🔢 miStudioScore Feature Scoring Workflow"
echo "========================================="

# 1. Health check
echo -e "\n1. 🔍 Checking service health..."
HEALTH_RESPONSE=$(curl -s "$BASE_URL/health")
echo "$HEALTH_RESPONSE" | jq '.'

SCORERS_AVAILABLE=$(echo "$HEALTH_RESPONSE" | jq -r '.available_scorers | length')
if [ "$SCORERS_AVAILABLE" -lt 2 ]; then
    echo "❌ Required scorers not available"
    exit 1
fi

echo "✅ Service healthy with $SCORERS_AVAILABLE scorers available"

# 2. Create comprehensive scoring configuration
echo -e "\n2. 📝 Creating scoring configuration..."
cat > config/production_scoring.yaml << 'EOF'
scoring_jobs:
  - scorer: "relevance_scorer"
    name: "ai_safety_score"
    params:
      positive_keywords: 
        - "safety"
        - "ethical"
        - "responsible"
        - "secure"
        - "privacy"
        - "transparent"
      negative_keywords:
        - "harmful"
        - "biased"
        - "dangerous"
        - "malicious"
        - "deceptive"
  
  - scorer: "relevance_scorer"
    name: "technical_complexity"
    params:
      positive_keywords:
        - "algorithm"
        - "optimization"
        - "architecture"
        - "performance"
        - "scalability"
        - "efficiency"
      negative_keywords:
        - "simple"
        - "basic"
        - "trivial"
        
  - scorer: "relevance_scorer"
    name: "business_value"
    params:
      positive_keywords:
        - "revenue"
        - "customer"
        - "profit"
        - "competitive"
        - "innovation"
        - "market"
      negative_keywords:
        - "cost"
        - "overhead"
        - "maintenance"
EOF

echo "✅ Scoring configuration created"

# 3. Create benchmark function for ablation scoring
echo -e "\n3. 🎯 Creating benchmark function..."
cat > config/production_benchmark.py << 'EOF'
import torch
import numpy as np

def run_benchmark(model, tokenizer, device):
    """
    Production benchmark measuring model performance on diverse tasks.
    """
    # Diverse test cases covering different capabilities
    test_cases = [
        "Explain the principles of machine learning in simple terms.",
        "What are the ethical considerations in AI development?",
        "Describe the process of neural network training.",
        "How does natural language processing work?",
        "What is the difference between supervised and unsupervised learning?",
        "Explain the concept of artificial general intelligence.",
        "What are the potential risks of AI technology?",
        "How can AI be used to solve climate change?",
        "Describe the role of data in machine learning.",
        "What is the future of artificial intelligence?"
    ]
    
    total_loss = 0.0
    total_perplexity = 0.0
    count = 0
    
    with torch.no_grad():
        for text in test_cases:
            try:
                # Tokenize input
                inputs = tokenizer(
                    text, 
                    return_tens