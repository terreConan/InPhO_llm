# InPhO LLM Research Project

Research infrastructure for comparing LLM responses to stratified human expertise levels in philosophical reasoning.

## Project Structure

```
InPhO_llm/
├── src/                          # Core infrastructure
│   ├── config.py                 # Configuration and version tracking
│   ├── api_manager.py           # LLM API management with caching
│   └── cost_analysis.py         # Cost estimation for different models
│
├── scripts/                      # Data processing scripts
│   ├── extract_expertise_levels.py    # Extract human data by expertise
│   ├── identify_comparison_pairs.py   # Find pairs with multi-expertise data
│   └── create_better_comparison_pairs.py  # Create coverage analysis
│
├── data/                         # All data files
│   ├── raw/                      # Original data files
│   │   ├── idea_evaluation_with_all_user_info.csv
│   │   ├── sep_idea_graph_edges.csv
│   │   ├── idea_id_label_mapping.csv
│   │   └── pairs.csv
│   │
│   ├── processed/                # Processed data files
│   │   ├── human_data_amateur.csv
│   │   ├── human_data_course_taker.csv
│   │   ├── human_data_phd_student.csv
│   │   ├── human_data_expert.csv
│   │   └── comparison_pairs_all_levels.csv
│   │
│   └── original/                 # Original researcher's data
│       ├── novice_ai_expert_human.csv
│       └── expert_ai_expert_human.csv
│
├── legacy/                       # Original research code
│   ├── relateAI.py
│   ├── merge_csv.py
│   └── agreement_matrix.py
│
├── docs/                         # Documentation
│   ├── memory-bank/              # Project documentation
│   └── TODO.txt
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Key Features

### Cost Analysis
- Compare 6 different LLM models
- Claude 3 Haiku: $0.94 for full dataset (vs $187.56 for Groq)
- Cost tracking and estimation tools

### Multi-Expertise Analysis
- Extract human data across all expertise levels (amateur to expert)
- 712 pairs with data in 2+ expertise levels

### API Management
- Caching and rate limiting
- Cost tracking
- Support for multiple LLM providers

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys:**
   - Copy `.env.example` to `.env` and fill in your keys
     ```bash
     cp .env.example .env
     ```
   - Or export in your shell profile:
     ```bash
     export GROQ_API_KEY="your_key_here"
     export ANTHROPIC_API_KEY="your_key_here"
     export GEMINI_API_KEY="your_key_here"
     export OPENAI_API_KEY="your_key_here"
     export MISTRAL_API_KEY="your_key_here"
     export GOOGLE_API_KEY="your_key_here"
     ```

3. **Run cost analysis:**
   ```bash
   python src/cost_analysis.py
   ```

4. **Extract expertise levels:**
   ```bash
   cd scripts
   python extract_expertise_levels.py
   ```

5. **Identify comparison pairs:**
   ```bash
   python identify_comparison_pairs.py
   ```

## Research Innovation

### Improvements
- Uses all expertise levels (amateur, course-taker, PhD, expert)
- Clean human baseline data (no AI contamination)
- 712 high-quality pairs with multi-expertise data
- 200x cost reduction with modern LLMs

## Next Steps

1. **Collect LLM responses** for the 712 comparison pairs
2. **Compare AI vs human judgments** across expertise levels
3. **Analyze systematic differences** in reasoning patterns
4. **Validate against original research** findings

## Collaboration

- **Andrew**: Working on Groq API responses for validation
- **You**: Multi-expertise analysis with modern LLMs
- **Complementary approaches**: Different models, different scopes 