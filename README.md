
## Features

- **Data Input**:
  - Upload CSV, XLS, or XLSX files
  - Use built-in example data for demonstration
  - Data preview and summary statistics

- **Statistical Analysis**:
  - One-Way ANOVA calculation
  - Effect size estimation (eta-squared)
  - Multiple significance level options (0.05, 0.01)

- **Post-hoc Tests**:
  - Tukey HSD (for equal variances, equal sample sizes)
  - Bonferroni (strict control of Type I error)
  - Scheffe (for all possible contrasts)
  - Games-Howell (for unequal variances)
  - Duncan (for higher statistical power)
  - Newman-Keuls (step-down procedure)

- **Assumption Checking**:
  - Shapiro-Wilk test for normality
  - Levene's test for homogeneity of variances
  - Alternative tests for when assumptions are violated:
    - Welch's ANOVA for heteroscedasticity
    - Kruskal-Wallis for non-normal data

- **Visualizations**:
  - Group boxplots
  - Bar plots with confidence intervals
  - Distribution plots
  - QQ plots for normality assessment

- **Results Export**:
  - Comprehensive Word document reports
  - CSV export of ANOVA tables

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/OneWay.git
cd OneWay

# Install required packages
pip install -r requirements.txt
```

## Requirements

```
streamlit
pandas
seaborn
numpy
scipy
statsmodels
matplotlib
python-docx
```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run Oneway.py
   ```

2. Open your web browser to the displayed URL (typically http://localhost:8501)

3. In the app:
   - Upload your data or use the example dataset
   - Select numeric and categorical columns for analysis
   - Run the ANOVA analysis
   - Explore the results, visualizations, and post-hoc tests
   - Download the complete report in Word format

## Workflow

The application follows a standard statistical workflow:

1. **Data Input**: Upload your data or use the provided example
2. **Assumption Checking**: Verify normality and homogeneity of variance
3. **ANOVA Analysis**: Calculate F-statistic and p-value
4. **Post-hoc Analysis**: If ANOVA is significant, determine which groups differ
5. **Visualization**: Explore data distributions and relationships
6. **Export Results**: Download comprehensive reports for documentation

## Example Use Cases

- Compare student performance across different teaching methods
- Analyze crop yields under different fertilizer treatments
- Evaluate customer satisfaction across different store locations
- Compare effectiveness of multiple drug treatments

## Additional Information

The "About ANOVA" tab provides comprehensive information about One-Way ANOVA, including:
- When to use ANOVA
- Key assumptions
- How to interpret results
- Formula explanations
- Academic references

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Developed by [Galuh Adi Insani](https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/)

---

*This README is for the One-Way ANOVA Analysis Tool, a statistical application built with Streamlit.*