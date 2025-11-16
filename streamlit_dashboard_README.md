# 📊 Streamlit Dashboard: Email Marketing Causal Analysis

An interactive dashboard showcasing comprehensive causal inference analysis for email marketing effectiveness.

## 🎯 Dashboard Overview

This Streamlit application provides an interactive interface to explore:
- The problem: Why naive analysis fails with confounding
- Multiple causal inference methods and their comparisons
- Treatment effect estimates and confidence intervals
- Subgroup analysis and heterogeneity
- Business impact and ROI calculations
- Robustness testing results

## 📑 Tab Structure

### **Tab 1: Overview** 🏠
- Project description and objectives
- Key findings summary
- Methodology overview
- Dataset information
- Quick metrics and highlights

### **Tab 2: The Problem** ⚠️
- Naive vs true effect comparison
- Bias visualization (68% overestimate!)
- Confounding evidence
  - Covariate imbalance (Love plots)
  - Standardized differences
  - Feature comparisons
- Who gets emails (demographics)
- Why naive analysis fails

### **Tab 3: Causal Methods** 🔬
- Method comparison table (6 methods)
- Detailed method descriptions
  - PSM (Propensity Score Matching)
  - DiD (Difference-in-Differences)
  - IPW (Inverse Probability Weighting)
  - AIPW (Doubly Robust)
  - T-Learner (Heterogeneous Effects)
- Bias comparison chart
- Method recommendations

### **Tab 4: Results** 📊
- Treatment effects with confidence intervals
- Subgroup analysis (RFM segments)
- Robustness summary
  - E-values
  - Placebo tests
  - Method agreement
- Business impact and ROI
- Summary metrics

## 🚀 Running the Dashboard

### Prerequisites
```bash
pip install -r requirements_streamlit.txt
```

### Run the App
```bash
streamlit run streamlit_app.py
```

Or with specific port:
```bash
streamlit run streamlit_app.py --server.port 8501
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Key Visualizations

### Interactive Charts
1. **Bias Visualization**: Naive vs True effect comparison
2. **Covariate Balance**: Standardized differences with thresholds
3. **Method Comparison**: Bar charts of estimates and bias
4. **Treatment Effects**: Estimates with 95% CI
5. **Subgroup Analysis**: Effects by RFM segment

### Metrics Dashboard
- Ground truth effect
- Sample size
- Treatment rate
- Methods compared
- Best estimate (PSM: 11.2%)
- Bias reduction (74%)
- Business impact (+$1.52M)

## 💡 Key Insights in Dashboard

### Statistical Findings
- **Naive Effect**: 16.0% (SEVERELY BIASED)
- **True Effect**: 9.5%
- **PSM Estimate**: 11.2% (Bias: 1.7 pp) ✅ BEST
- **Method Cluster**: 11-14% across valid methods
- **Heterogeneity**: 9.0% (Low RFM) to 18.6% (Loyal)

### Business Insights
- **ROI Range**: 49,922% - 103,677%
- **Best Segments**: Loyal (Q4), Medium RFM (8-10)
- **Optimal Strategy**: Email 81.7% of customers
- **Expected Impact**: +$1.52M profit (+21.7%)
- **Recommendation**: Personalize, don't exclude

## 🎨 Design Features

### Professional Styling
- Custom CSS with branded colors
- Metric cards with highlights
- Color-coded validity indicators (✅/❌)
- Expandable sections for details
- Responsive layout

### Navigation
- Sidebar navigation between tabs
- Clear section headers
- Intuitive flow: Problem → Methods → Results
- Progress indicators

### Data Presentation
- Interactive Plotly charts
- Sortable DataFrames
- Expandable details
- Tooltips for context
- Metric comparisons

## 📁 Files

```
├── streamlit_app.py                    # Main dashboard application
├── requirements_streamlit.txt          # Dependencies
└── streamlit_dashboard_README.md      # This file
```

## 🔧 Technical Details

### Dependencies
- `streamlit>=1.28.0` - Web app framework
- `plotly>=5.17.0` - Interactive visualizations
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.20.0` - Numerical computing

### Data Sources
- `data/processed/data_with_propensity_scores.csv`
- `data/processed/ground_truth.json`

### Cached Operations
- Data loading is cached for performance
- Interactive visualizations with Plotly
- Real-time calculations

## 📖 Usage Guide

### For Executives
1. Start at **Overview** tab
2. Review key metrics
3. Check business impact
4. Read recommendations

### For Data Scientists
1. Go to **The Problem** tab
2. Understand confounding
3. Review **Methods** tab
4. Compare approaches
5. Analyze **Results** in detail

### For Marketing Teams
1. **Overview**: Key findings
2. **Results**: Subgroup analysis
3. Business impact and ROI
4. Targeting recommendations

## 🎓 Educational Value

This dashboard demonstrates:
- **Causal Inference**: From problem to solution
- **Confounding**: Why naive analysis fails
- **Multiple Methods**: PSM, DiD, IPW, AIPW, T-Learner
- **Robustness Testing**: E-values, placebo tests
- **Business Translation**: ROI and strategy
- **Data Visualization**: Interactive charts

## 🔍 Key Findings Highlighted

### In Dashboard
1. **Naive = 16.0%** but True = 9.5% (68% bias!)
2. **PSM = 11.2%** closest to truth
3. **Medium RFM = 17.1%** uplift (best segment)
4. **Loyal customers = 18.6%** uplift
5. **Email all 81.7%** customers (optimal)
6. **ROI = 50K-104K%** (astronomical!)

### Color Coding
- 🟢 Green = Valid methods / Good results
- 🔴 Red = Invalid methods / Concerns
- 🟡 Yellow = Warnings / Moderation needed
- 🔵 Blue = Neutral / Information

## 🚀 Deployment

### Local Development
```bash
streamlit run streamlit_app.py
```

### Cloud Deployment (Streamlit Community Cloud)
1. Push code to GitHub
2. Visit share.streamlit.io
3. Connect your repo
4. Deploy instantly

### Docker Deployment
```dockerfile
FROM python:3.9
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

## 📞 Support

For questions about:
- **Statistical Methods**: Review implementation in `src/causal/`
- **Data**: Check `data/processed/` directory
- **Business Insights**: See `BUSINESS_ANALYSIS_SUMMARY.md`
- **Robustness**: Review `ROBUSTNESS_ANALYSIS_SUMMARY.md`

## 🎯 Next Steps

1. **Explore** all 4 tabs
2. **Interact** with visualizations
3. **Review** method comparisons
4. **Check** business recommendations
5. **Validate** with real data

---

**Built with Streamlit | Powered by Causal Inference | Designed for Decision Makers**

Generated: 2025-11-16
Project: Causal Impact of Email Marketing on Purchase Behavior
