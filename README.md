# Video Aggression Detection Benchmark

Comprehensive benchmark dataset and evaluation pipeline for vision-language models (VLMs) on video aggression detection tasks.

## Overview

This benchmark evaluates VLMs on their ability to understand aggressive behavior in videos through multiple-choice question answering. The pipeline consists of two main stages:

1. **Question Generation**: Create diverse, multi-difficulty questions from video annotations
2. **Evaluation**: Run VLMs on generated questions and compute accuracy metrics

## Dataset Structure

### Input Files

- `annotations.json`: Video annotations containing:
  - `file_name`: Video filename
  - `aggressor`: List of people displaying aggressive behavior
  - `victim`: List of people being victimized
  - `bystanders`: List of bystanders
  - `action`: Primary aggressive action
  - `environment`: Scene location/context

### Question Types (14 total)

Questions are organized into **3 difficulty tiers** plus secondary types:

#### Basic Questions (4 types)
- **Primary Action**: "What aggressive action takes place?"
- **Aggressor Identification**: "Who displays aggressive behavior?"
- **Victim Recognition**: "Who is victimized?"
- **Role Identification**: "What role does this person play?"

#### Compound Questions (3 types)
- **Compound Action+Victims**: "What action is performed and who is victimized?"
- **Compound Action+Aggressor**: "What action is performed and who performs it?"
- **Compound Aggressor+Victim**: "Who aggresses against whom?"

#### Detailed Questions (2 types)
- **Compound Aggressor+Action+Victim**: "Who performs what action on whom?" (includes frequency-inverted distractors)
- **Sequence Verification**: "Verify if action sequence is correct" (includes frequency-inverted distractors)

#### Secondary Questions (5 types)
- **Compound Action+Location**: "What action occurs and where?"
- **Compound Aggressor+Victim+Count**: "Count of aggressors and victims?"
- **Role Count Aggressor**: "How many people display aggressive behavior?"
- **Role Count Victim**: "How many people are victimized?"
- **Role Count Bystander**: "How many bystanders are present?"

### Distribution

- **Primary Questions**: 15,004 questions across 9 types
- **Secondary Questions**: 3,744 questions across 5 types
- **Total**: 18,748 generated questions from 2,670 videos

See `prompt_generator/templates.py` for the authoritative definition of `SECONDARY_QUESTION_TYPES`.

## Question Generation

### Quick Start

Generate questions locally (no GPU required):

```bash
# Basic usage
python generate_questions_local.py annotations.json

# With custom output file
python generate_questions_local.py annotations.json -o generated_questions.json

# Sample a subset (5% of videos)
python generate_questions_local.py annotations.json --sample 0.05 --seed 42

# Sample specific count
python generate_questions_local.py annotations.json --sample 100 --seed 42
```

### Parameters

- `annotations.json`: Input annotation file (required)
- `-o, --output`: Output file path (default: `generated_questions.json`)
- `--sample`: Fraction (0-1) or count of videos to sample (default: all)
- `--seed`: Random seed for reproducibility (default: none)
- `-d, --depth`: Max difficulty for distractors (default: 8)
- `--recipes`: Custom hardness recipe JSON file (optional)

### Output Format

The script generates `generated_questions.json`:

```json
{
  "metadata": {
    "generated_at": "2025-05-04T12:34:56.789Z",
    "total_questions": 18748,
    "total_videos": 2674,
    "distribution": {
      "simple": 5,
      "compound": 6,
      "complex": 3,
      "counting": 3,
      "identification": 4
    },
    "hardness_profile": "frequency_inverted"
  },
  "questions_by_video": {
    "video_001.mp4": [
      {
        "video_name": "video_001.mp4",
        "question_type": "primary_action",
        "category": "simple",
        "prompt": "What aggressive action takes place in this video?",
        "answers": [
          "Pushing",
          "Pulling hair",
          "Kicking",
          "Punching"
        ],
        "correct_index": 0,
        "correct_answer": "Pushing",
        "hardness": "none_claim"
      }
    ]
  }
}
```

### How Questions Are Generated

1. **Template Selection**: For each video, the `CategoryDistributor` selects 5 question types (one per category) ensuring no duplicates
2. **Answer Construction**: Correct answers are built from video annotations
3. **Distractor Generation**: Up to 7 distractors per question (3 for role identification) using hardness strategies:
   - **role_reversal**: Swap aggressor/victim
   - **wrong_action**: Use action from different video
   - **wrong_victim/aggressor**: Use wrong role from same video
   - **cross_video**: Use role/action from different video
   - **bystander_substitution**: Replace role with bystander
   - **frequency_saturation** (complex only): Balance person/action frequencies

4. **Hardness Profiles**: Each question type has a recipe defining distractor composition


### Local Evaluation (Development)

For single-GPU testing on your machine:

```bash
python -m prompt_generator.evaluation.run_evaluation \
  annotations.json \
  /path/to/videos \
  --model "Qwen/Qwen2.5-VL-7B-Instruct" \
  --num-questions 10 \
  --output-dir ./results
```

### Evaluation Arguments

Common parameters for `run_evaluation.py`:

```bash
python -m prompt_generator.evaluation.run_evaluation \
  annotations.json \
  video_dir \
  --model MODEL_PATH              # Hugging Face model path
  --conda-env ENV_NAME            # Conda environment (server only)
  --num-questions N               # Number of questions (default: 10)
  --num-frames K                  # Frames per video (default: 8)
  --batch-size B                  # Batch size for inference (default: 1)
  --device DEVICE                 # 'cuda' (default) or 'cpu'
  --output-dir DIR                # Output directory (default: '.')
  --output-csv results.csv        # Export results to CSV
  --checkpoint                    # Enable checkpointing for long runs
  --part N --total-parts M        # Process part N of M (for splitting large jobs)
```

### Evaluation Output Format

Results are saved to `evaluation_results_<timestamp>.json`:

```json
{
  "metadata": {
    "model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
    "num_frames": 8,
    "timestamp": "2025-05-04T12:34:56.789Z",
    "total_questions": 100
  },
  "summary": {
    "total_questions": 100,
    "correct_count": 82,
    "accuracy": 0.82,
    "accuracy_by_type": {
      "primary_action": {
        "total": 10,
        "correct": 9,
        "accuracy": 0.90
      },
      "compound_aggressor_action_victim": {
        "total": 8,
        "correct": 6,
        "accuracy": 0.75
      }
    }
  },
  "results": [
    {
      "video_name": "video_001.mp4",
      "question_type": "primary_action",
      "prompt": "What aggressive action takes place?",
      "answers": ["Pushing", "Pulling", "Kicking", "Punching"],
      "correct_answer": "Pushing",
      "correct_index": 0,
      "model_response": "Pushing",
      "model_selected_index": 0,
      "is_correct": true
    }
  ]
}
```

### Interpreting Results

**Accuracy by Question Type**: Compare model performance across question difficulties to identify weak areas:
- Basic questions
- Compound questions
- Detailed questions

**Primary vs Secondary Split**: Analyze results separately:
- Extract questions where `question_type` is in `SECONDARY_QUESTION_TYPES`

## Data Format Specification

### Annotations Format

Expected JSON structure:

```json
[
  {
    "file_name": "video_001.mp4",
    "aggressor": ["Person A", "Person B"],
    "victim": ["Person C"],
    "bystanders": ["Person D"],
    "action": "Pushing",
    "environment": "School hallway"
  }
]
```

All fields are optional for flexibility:
- Missing `aggressor` → answers will say "No one displays aggressive behavior"
- Missing `victim` → answers will say "No one appears to be victimized"
- Missing `environment` → answers will say "location unclear"

### Generated Questions Format

Each question object contains:

```json
{
  "video_name": "string",
  "question_type": "string (one of 14 types)",
  "category": "string (basic|compound|detailed|secondary)",
  "prompt": "string (the question text)",
  "answers": ["string", "string", "string", "string"],
  "correct_index": 0,
  "correct_answer": "string",
  "option_hardness": ["string", "string", "...(one per answer option)"]
}
```

### Evaluation Results Format

Each result object contains:

```json
{
  "video_name": "string",
  "question_type": "string",
  "prompt": "string",
  "answers": ["string", "string", "string", "string"],
  "correct_answer": "string",
  "correct_index": 0,
  "model_response": "string (raw model output)",
  "model_selected_index": 0,
  "is_correct": true
}
```