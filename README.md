# NFL Rushing Predictor 🏈

A machine learning project to predict the 2025 NFL rushing leader using historical data and advanced feature engineering.


## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd nfl_rushing_predictor
```

2. **Create and activate virtual environment**
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create necessary directories**
```bash
mkdir -p data/raw data/processed models logs
```

## 📊 Usage

### Basic Usage

Run the predictor with default settings:

```bash
python run_predictor.py
```

### Advanced Usage

```bash
# Use different model types
python run_predictor.py --model-type random_forest

# Adjust blending method
python run_predictor.py --blend-method sigmoid --blend-exponent 1.5

# Apply top-N uplift
python run_predictor.py --top-n 5 --top-n-multiplier 1.15

# Custom output location
python run_predictor.py --output-dir ./results --output-file my_predictions.csv

# Save the trained model
python run_predictor.py --save-model

# Verbose logging
python run_predictor.py --verbose

# Disable caching (fetch fresh data)
python run_predictor.py --no-cache
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-type` | `gradient_boosting` | Model type (gradient_boosting, random_forest, ridge, lasso) |
| `--blend-method` | `sigmoid` | Blending method (linear, power, sigmoid, none) |
| `--blend-exponent` | `1.0` | Exponent for power/sigmoid blending |
| `--top-n` | `3` | Number of top players to uplift |
| `--top-n-multiplier` | `1.1` | Multiplier for top-N uplift |
| `--no-cache` | `False` | Disable data caching |
| `--output-dir` | `.` | Output directory for predictions |
| `--output-file` | `predictions_2025.csv` | Output filename |
| `--display-top-n` | `10` | Number of top predictions to display |
| `--save-model` | `False` | Save trained model to disk |
| `--log-level` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `--verbose` | `False` | Enable verbose output (DEBUG level) |

## 📁 Project Structure

```
nfl_rushing_predictor/
├── config_settings.py      # Configuration and settings
├── data_loader.py          # Data loading with rate limiting
├── data_processor.py       # Feature engineering and preprocessing
├── predictor_model.py      # ML model implementation
├── run_predictor.py        # Main execution script
├── utils_helpers.py        # Utility functions
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── .env.example           # Environment variables template
├── README.md              # This file
├── LICENSE                # Project license
├── data/                  # Data directory (gitignored)
│   ├── raw/              # Raw data cache
│   ├── processed/        # Processed data
│   └── cache/            # Temporary cache
├── models/                # Saved models (gitignored)
└── logs/                  # Log files (gitignored)
```

## 🔒 Security Considerations

### Environment Variables

Create a `.env` file for any API keys or sensitive configuration:

```bash
# Example .env file
NFL_API_KEY=your_api_key_here
PRO_FOOTBALL_REF_API_KEY=your_api_key_here
```

**Never commit `.env` to version control!**

### Rate Limiting Configuration

Adjust rate limits in `config_settings.py`:

```python
API_RATE_LIMITS = {
    'requests_per_minute': 10,  # Conservative limit
    'requests_per_hour': 100,
    'backoff_factor': 2.0,
}

PFR_CONFIG = {
    'request_delay': 3.0,  # Seconds between requests
    'max_retries': 2,
    'timeout': 10,
}
```

## 🧪 Testing

Run tests (when available):

```bash
pytest tests/
```

## 📈 Model Performance

The model uses gradient boosting by default with the following features:
- Age and career stage metrics
- Historical performance trends
- Team offensive line quality
- Injury history and durability
- Usage patterns and role indicators
- Era-specific adjustments

Typical performance metrics:
- MAE (Mean Absolute Error): ~150-200 yards
- R² Score: ~0.65-0.75

## 🛠️ Troubleshooting

### Common Issues

1. **Module not found errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Permission errors on data directory**
   ```bash
   chmod -R 755 data/ models/ logs/
   ```

3. **Rate limit errors**
   - Increase delay in `config_settings.py`
   - Use cached data with default settings
   - Reduce `requests_per_minute` limit

4. **Memory errors with large datasets**
   - Use `--min-samples` to limit training data
   - Process data in smaller chunks

## 📝 Configuration

Key configuration options in `config_settings.py`:

```python
# Data range
START_YEAR = 2000
END_YEAR = 2024
CURRENT_SEASON = 2025

# Model hyperparameters
MODEL_CONFIG = {
    'n_estimators': 300,
    'learning_rate': 0.1,
    'max_depth': 6,
    'random_state': 42,
}

# Feature engineering
FEATURE_CONFIG = {
    'prime_age_range': (24, 28),
    'high_volume_threshold': 250,
    # ... more settings
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style

This project follows:
- PEP 8 style guidelines
- Type hints where applicable
- Comprehensive docstrings
- Error handling best practices

Run code formatters:
```bash
black *.py
flake8 *.py
```

## 📄 License

See `LICENSE` file for details.

## ⚠️ Disclaimer

This project is for educational and entertainment purposes only. Predictions are based on historical data and statistical models, and should not be used for gambling or financial decisions.

The model's predictions are approximations and may not reflect actual 2025 season outcomes due to:
- Injuries
- Trades and roster changes
- Coaching changes
- Unpredictable game situations
- Rule changes

## 🙏 Acknowledgments

- Pro Football Reference for historical data
- NFL community for inspiration
- scikit-learn for machine learning tools

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues and discussions
- Review the troubleshooting section

---

**Last Updated**: February 2025

**Version**: 1.0.0
