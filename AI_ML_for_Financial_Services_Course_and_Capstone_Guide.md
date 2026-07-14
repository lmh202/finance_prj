# AI/ML for Financial Services

## Course Overview and Capstone Project Requirements

**Course code:** SWS3022 / 2026_SWS3022  
**Course title:** AI/ML for Financial Services  
**Format:** Intensive, hands-on workshop with individual exercises and a team capstone project  
**Primary environment:** Python, Jupyter Notebook, and data-driven financial applications

> This guide consolidates the current Canvas course page, assignment information, and the available lecture materials. Where an older slide and the current Canvas course page differ, the current Canvas course page or assignment page should be treated as authoritative. Students should still check Canvas announcements and assignment pages for final dates and instructions.

---

## 1. Course Purpose

AI/ML for Financial Services introduces the practical use of artificial intelligence, machine learning, data analytics, and software systems in finance. The course is designed around a complete applied workflow:

**financial data -> data cleaning and feature engineering -> AI/ML or trading models -> financial and risk evaluation -> GenAI/NLP applications -> fintech dashboard or product prototype**

The course covers applications in:

- Algorithmic trading and investment analytics
- Portfolio analysis and risk management
- Credit risk and default prediction
- Fraud, money laundering, and market-manipulation detection
- Financial news, reports, sentiment, and document intelligence
- Generative AI, large language models, and retrieval-augmented generation (RAG)
- Customer-facing fintech applications and decision-support tools
- Responsible AI, governance, and model-risk assessment

The emphasis is not only on predictive accuracy. Students are expected to connect technical results to financial meaning, compare models or strategies against appropriate benchmarks, and communicate limitations responsibly.

---

## 2. Prerequisites

Students should be comfortable with basic Python programming, including:

- Variables, functions, and control flow
- Lists, dictionaries, and other basic data structures
- Introductory NumPy and Pandas operations
- Basic use of Jupyter Notebook

Prior finance knowledge is not required. Financial concepts are introduced during the course, but students are expected to learn and apply them actively.

### Pre-course technical setup

The course materials recommend:

1. Install the Anaconda Distribution with a recent Python 3 version.
2. Launch Jupyter Notebook or JupyterLab through Anaconda Navigator.
3. Install or update the main packages:

   ```bash
   pip install -U numpy pandas matplotlib seaborn jupyter notebook openpyxl
   ```

4. Verify that NumPy, Pandas, Matplotlib, Seaborn, and OpenPyXL import correctly.
5. Create a working folder such as:

   ```text
   SWS3022FinTech/
   |-- notebooks/
   |-- data/
   `-- outputs/
   ```

Google Colab may be used as a temporary backup, but the main teaching environment is Anaconda with Jupyter Notebook.

---

## 3. Learning Outcomes

By the end of the workshop, students should be able to:

1. Explain major AI/ML use cases across the financial-services value chain.
2. Retrieve, clean, visualize, and interpret financial market data.
3. Engineer features such as returns, moving averages, volatility, momentum, and correlations.
4. Implement and backtest basic algorithmic trading strategies.
5. Calculate and interpret volatility, Sharpe ratio, beta, alpha, drawdown, and other risk metrics.
6. Train and evaluate machine-learning models for financial prediction and decision support.
7. Distinguish technical model metrics from financial performance metrics.
8. Explain how NLP, GenAI, and RAG can be applied to financial text and documents.
9. Build a usable financial analytics dashboard or lightweight fintech application.
10. Evaluate financial AI systems using responsible-AI and model-risk considerations.
11. Design, implement, and present a team-based fintech prototype.

---

## 4. Main Technologies and Methods

### Programming and data tools

- Python and Jupyter Notebook
- NumPy and Pandas
- Matplotlib and Seaborn
- yfinance and selected alternative financial-data sources
- scikit-learn
- TensorFlow/Keras and selected deep-learning concepts
- QuantConnect for algorithmic trading and backtesting
- Streamlit for rapid dashboard development
- Optional full-stack path using a frontend, API, and database

### Financial analytics

- Prices, returns, and cumulative returns
- Moving averages and Bollinger Bands
- Momentum and volatility
- Correlation and diversification
- Beta and alpha
- Sharpe, Sortino, and Treynor ratios
- Maximum drawdown
- Benchmark comparison
- Stop-loss, take-profit, and portfolio rebalancing

### AI/ML topics

- Supervised learning: classification and regression
- Logistic regression, decision trees, random forests, SVM, and related models
- Unsupervised learning: clustering and dimensionality reduction
- Neural networks, recurrent neural networks, and LSTM
- Time-series methods such as ARIMA and GARCH
- Train/test splits, time-series validation, hyperparameter tuning, and ensemble learning
- Imbalanced-data handling
- ROC/AUC, confusion matrix, precision, recall, and F1 score
- NLP, sentiment analysis, topic modelling, transformers, and LLMs
- GenAI, financial document Q&A, and RAG

---

## 5. Teaching Structure and Course Sequence

The first six instructional days combine a morning lecture, explanation, live demonstration, and short exercise with an afternoon hands-on laboratory. After these instructional days, the course shifts mainly to team-project development, consultation, integration, evaluation, and presentation.

| Stage | Main topics | Expected output |
|---|---|---|
| Day 0: Course preview | Financial-services value chain, AI/ML use cases, Jupyter, NumPy, Pandas, Matplotlib, yfinance, financial time series | Optional HW0 and technical setup |
| Day 1: Financial data analytics | Prices, returns, moving averages, rolling volatility, cumulative return, correlation, problem framing | HW1 and initial project-track exploration |
| Day 2: Algorithmic trading | Market structure, strategy logic, QuantConnect, buy-and-hold, moving-average strategies, backtesting pitfalls | HW2 and QuantConnect Bootcamp work |
| Day 3: Risk and strategy evaluation | Beta, alpha, volatility, Sharpe ratio, correlation, diversification, drawdown, benchmark comparison | HW3 strategy and risk analysis |
| Day 4: Machine learning in finance | Feature engineering, temporal train/test split, classification, random forest, ML metrics versus financial metrics | ML or deep-learning exercise/HW4 |
| Day 5: Financial text intelligence | NLP, sentiment, LLMs, GenAI, RAG, hallucination risk, human review, responsible AI | Financial text mini-project/HW5 |
| Day 6: Productization and project launch | Streamlit or full-stack architecture, project scoping, data plan, model plan, dashboard plan | Team project proposal |
| Day 7: Data validation | Data feasibility, cleaning, and baseline design | Confirmed data source and baseline plan |
| Day 8: Features and baseline | Feature engineering, target definition, baseline model or strategy | Initial results |
| Day 9: Improvement and dashboard skeleton | Model/strategy improvement, risk evaluation, dashboard structure | Prototype application skeleton |
| Day 10: Progress presentation | Problem, user, data, method, prototype, preliminary results, difficulties | Progress presentation and feedback |
| Day 11: Implementation and integration | Debugging and end-to-end integration | Working end-to-end prototype |
| Day 12: Evaluation and refinement | Metrics, benchmark comparison, limitations, responsible AI | Final evaluation results |
| Day 13: Product and presentation polish | Dashboard usability, slides, poster, rehearsal | Final presentation draft |
| Day 14: Final presentation | Financial problem, pipeline, model/strategy, evaluation, application demo, risks and limitations | Final project submission |
| Day 15: Showcase | Poster/demo session, peer learning, feedback, and reflection | Showcase participation |

Students are also expected to complete introductory QuantConnect Bootcamp lessons to become familiar with the platform and its algorithmic-trading workflow.

---

## 6. Assessment

| Component | Weight | Description |
|---|---:|---|
| Individual quizzes, exercises, and homework | 25% | Short individual tasks completed during lectures and practical sessions |
| Group capstone project | 75% | Design, implementation, evaluation, and presentation of an AI/ML fintech prototype |

The capstone project is completed in teams of four. Peer evaluation will be used, and individual marks may be adjusted to reflect each member's contribution.

---

# Capstone Project Requirements

## 7. Project Objective

Each team must design, build, evaluate, and present an AI/ML-enabled fintech prototype. A strong project should demonstrate an end-to-end connection between:

1. A meaningful financial problem
2. A clearly defined user or stakeholder
3. A feasible data source
4. A reproducible data pipeline
5. Appropriate financial features
6. A valid model, strategy, or analytical method
7. Technical and financial evaluation
8. A usable dashboard or application
9. Responsible-AI, model-risk, and limitation analysis

The project should be more than a standalone prediction notebook. It should communicate why the problem matters in finance, how the result would be used, and what risks would arise if the system were deployed.

---

## 8. Team Requirements

- **Team size:** Four students per team.
- **Contribution:** Each member should have an identifiable role and substantive contribution.
- **Peer evaluation:** Individual grades may be adjusted based on peer feedback and demonstrated contribution.
- **Current Canvas group:** Project Group 8.

An older Lecture 1 slide also states that no more than two members should come from the same university. Because this condition is not repeated on the current Canvas course page, the team should confirm it with the teaching staff if relevant.

---

## 9. Recommended Project Scope

A suitable project should be achievable within the workshop period and should have:

- Accessible data that can be obtained legally and reliably
- A manageable financial question rather than an overly broad platform idea
- A simple baseline that can be completed early
- At least one meaningful model or strategy improvement
- A benchmark for comparison
- Measurable technical and financial outcomes
- A demonstrable interface
- A clear discussion of assumptions, limitations, and risks

Projects should avoid depending on unavailable proprietary data, production-level brokerage integration, or an unrealistically large engineering scope.

---

## 10. Suggested Project Themes

The course materials list or illustrate the following directions:

### Trading and investment

- Algorithmic trading using technical, fundamental, or ML signals
- Multi-asset or multi-strategy portfolio analysis
- Portfolio optimization and risk-aware allocation
- Robo-advisory based on user risk profiles
- Value-investing or stock-screening assistants
- Short-squeeze or unusual-market-activity detection
- Foreign-exchange or virtual-asset analytics

### Credit and institutional risk

- Loan-default prediction
- Credit scoring and borrower-risk analysis
- Corporate bankruptcy prediction
- ESG-informed credit-risk assessment
- Fraud detection
- Anti-money-laundering analytics
- Market-manipulation detection

### Financial intelligence and GenAI

- Financial-news sentiment analysis
- Annual-report or earnings-call intelligence
- Social-media signals for investment research
- RAG-based financial document Q&A
- Financial research copilots
- Personal financial assistants or virtual relationship managers

### Consumer and educational fintech

- Gamified investing or financial-literacy applications
- Risk-profile and portfolio-learning tools
- Remittance-service comparison tools
- Interactive dashboards for financial decision support

Past examples shown in the lecture materials include multi-agent stock-investment suggestion systems, personal AI financial assistants, real-time investment platforms, gamified stock-market learning, market-manipulation simulations, virtual-asset prediction platforms, and foreign-exchange copilots.

---

## 11. Required Project Proposal

The project-launch stage requires a proposal in Markdown or PDF format. The proposal should contain at least:

1. **Project title**
2. **Team members and responsibilities**
3. **Financial problem and motivation**
4. **Target user or stakeholder**
5. **Research question or product objective**
6. **Data source, access method, coverage, and licensing considerations**
7. **Planned data-cleaning and feature-engineering steps**
8. **Baseline model, strategy, or heuristic**
9. **Proposed AI/ML model or trading strategy**
10. **Benchmark and evaluation metrics**
11. **Dashboard or application plan**
12. **Responsible-AI and model-risk considerations**
13. **Expected limitations**
14. **Development milestones**

The current lecture material asks teams to finalize their topic by **Thursday, 16 July 2026**. The team should verify the precise submission mechanism and time on Canvas.

---

## 12. Data and Feature Engineering Requirements

Teams should explain:

- Why the selected data is relevant to the financial problem
- The time period, sampling frequency, assets, firms, users, or documents covered
- Missing values, outliers, survivorship bias, selection bias, and data leakage
- How raw variables are transformed into meaningful financial features
- Whether transaction costs, slippage, delays, or other realistic constraints are included
- How reproducibility is maintained

Possible features include returns, lagged returns, rolling volatility, moving averages, momentum, volume, valuation ratios, macroeconomic variables, sentiment scores, embeddings, user-risk indicators, or domain-specific features.

For time-series projects, random train/test splitting should generally be avoided. Training data must precede validation and test data to prevent the model from seeing future information.

---

## 13. Model or Strategy Requirements

The selected method must match the problem. Examples include:

- Classification for direction, default, fraud, or event prediction
- Regression for returns, volatility, prices, or risk estimates
- Clustering for customer, asset, or behavior segmentation
- NLP models for sentiment, topic discovery, or document classification
- RAG systems for document-grounded financial question answering
- Trading strategies based on technical, fundamental, or ML signals
- Portfolio models for allocation and risk management

Teams should include a simple baseline before introducing a more complex model. Complexity alone is not evidence of quality. The final report should explain the model logic, assumptions, hyperparameters, and major failure modes.

---

## 14. Evaluation Requirements

Evaluation should include both technical and financial perspectives where applicable.

### Technical metrics

- Accuracy, precision, recall, F1 score
- ROC/AUC
- Confusion matrix
- MAE, RMSE, or other regression errors
- Stability across time periods or validation folds
- Retrieval quality and groundedness for RAG applications

### Financial metrics

- Cumulative and annualized return
- Volatility
- Sharpe or Sortino ratio
- Beta and alpha
- Maximum drawdown
- Correlation with a benchmark
- Turnover and transaction-cost sensitivity
- Diversification and concentration

### Benchmarking

Projects must compare their output with an appropriate reference, such as:

- Buy-and-hold
- A market index
- Equal-weight allocation
- A simple moving-average rule
- A majority-class or naive prediction baseline
- A standard statistical model
- A non-RAG or simpler information-retrieval baseline

Teams should report negative or inconclusive results honestly. A well-designed evaluation and clear explanation can be more valuable than an apparently high score produced by leakage or unrealistic assumptions.

---

## 15. Dashboard or Application Requirements

The course expects a usable interface, normally implemented with Streamlit or an approved alternative architecture. The application should help a user understand or act on the project's results.

The interface should ideally include:

- Clear input controls or data selection
- Key financial outputs and visualizations
- Model or strategy results
- Benchmark comparison
- Risk indicators and warnings
- Explanations suitable for the intended user
- Limitations and responsible-use notices

The dashboard does not need production-level infrastructure, but it should work reliably for the final demonstration.

---

## 16. Responsible AI and Model Risk

Every project should discuss:

- Data quality, bias, and representativeness
- Financial harm caused by false positives or false negatives
- Explainability and human oversight
- Privacy, confidentiality, and licensing
- Hallucination and unsupported claims in LLM or RAG systems
- Distribution shift and changing market regimes
- Overfitting, data leakage, and backtest over-optimization
- Misleading certainty in predictions or recommendations
- Appropriate disclaimers and intended-use boundaries

Teams should avoid presenting the prototype as guaranteed financial advice or as a production-ready trading system.

---

## 17. Capstone Grading Rubric

The current Canvas course page provides the following rubric:

| Criterion | Weight |
|---|---:|
| Financial relevance and problem framing | 20% |
| Data preparation and feature engineering | 20% |
| Model or strategy correctness | 20% |
| Evaluation quality and benchmark comparison | 20% |
| Dashboard or application usability | 10% |
| Responsible AI and model-risk discussion | 5% |
| Presentation quality | 5% |
| **Total** | **100%** |

An older Lecture 1 slide uses a different breakdown: innovative idea 10%, data selection and preparation 10%, AI/ML/statistical modelling 60%, and web application 20%. Unless the instructor states otherwise, the current Canvas rubric above should be followed.

---

## 18. Project Milestones

| Milestone | Expected evidence |
|---|---|
| Topic selection | Clear financial problem, user, and feasible scope |
| Proposal | Data, features, method, benchmark, dashboard, metrics, risks, and timeline |
| Data validation | Working data loader and documented cleaning process |
| Baseline | Reproducible baseline notebook or strategy |
| First model/strategy | Initial results with valid temporal or cross-validation design |
| Dashboard skeleton | Working application structure connected to sample results |
| Progress presentation | Problem, data, method, prototype, results, difficulties, and next steps |
| Integrated prototype | End-to-end data, model, evaluation, and interface workflow |
| Final evaluation | Benchmark comparison, risk metrics, limitations, and responsible-AI discussion |
| Final presentation | Polished demo, slides, poster, and concise narrative |

---

## 19. Final Deliverables

Each team should submit:

1. Final presentation slides
2. Project poster
3. Code, notebooks, repository, or a zipped project folder
4. Dashboard/application link or screenshots
5. A short README or project report covering:
   - Problem statement
   - Target user
   - Data source
   - Method
   - Feature engineering
   - Evaluation results
   - Benchmark comparison
   - Limitations
   - Responsible-AI and model-risk considerations
6. A short ACM/IEEE-style research paper if applicable

A recommended project package is:

```text
project-name/
|-- README.md
|-- requirements.txt
|-- data/                 # only distributable or sample data
|-- notebooks/
|-- src/
|-- app/
|-- outputs/
|-- slides/
|-- poster/
`-- report/
```

---

## 20. Presentation Expectations

The final presentation should tell a coherent story:

1. What financial problem is being solved?
2. Who needs the solution and why?
3. What data is used?
4. How is the data processed and transformed?
5. What model or strategy is implemented?
6. What is the benchmark?
7. What do the technical and financial results show?
8. How does the dashboard support the target user?
9. What can go wrong?
10. What would be improved with more time or better data?

The demonstration should prioritize a stable, understandable end-to-end workflow over a large number of unfinished features.

---

## 21. Academic Integrity and Use of AI Tools

The current Canvas assignment policy states that AI tools are generally permitted for take-home work, provided their use is clearly acknowledged. Failure to declare AI use may be treated as plagiarism. Students remain responsible for the accuracy, quality, originality, and compliance of everything submitted.

A suitable declaration format is:

> I used [AI tool name] to [describe the specific uses, such as brainstorming, code explanation, debugging, editing, visualization, or drafting]. I reviewed and verified the resulting content and remain responsible for the quality and accuracy of the submitted work.

Teams should also keep a short record of important prompts, generated code, external data, and third-party libraries when these materially affect the project.

---

## 22. Project Selection Checklist

Before committing to a topic, the team should be able to answer **yes** to most of the following:

- Is the problem clearly financial?
- Is there a specific user or stakeholder?
- Can the necessary data be obtained immediately?
- Can a baseline be completed within the first project-studio sessions?
- Is there a defensible model, strategy, or analytical method?
- Is there an appropriate benchmark?
- Can success be measured with financial as well as technical metrics?
- Can the result be demonstrated through a dashboard or application?
- Can the team explain risks, limitations, and responsible use?
- Can four members make distinct, meaningful contributions?

If the answer to several of these questions is no, the topic should be narrowed or changed before implementation begins.

---

## 23. Current Course Notes

- The current Canvas assignment page lists **Homework 3** as due on **14 July 2026 at 23:59 Singapore time**. The assignment accepts ZIP, PDF, or IPYNB files.
- Lecture 3 describes HW3 as a strategy-design task involving at least three U.S. stocks, more than five years of backtesting, diversification, benchmark correlation, positive returns, and improvement of risk-adjusted performance.
- Some slide deadlines differ from the live Canvas assignment page. Live Canvas dates should be treated as authoritative.
- Course materials are provided for personal educational use and should not be redistributed without permission.

---

## 24. Summary

The course is designed to move from basic financial-data analysis to a complete fintech prototype. The capstone project is the central assessment and should integrate finance, data engineering, AI/ML or strategy design, evaluation, user experience, and responsible deployment considerations.

The strongest project will not necessarily use the most complex model. It will define a valuable financial problem, use reliable data, implement a correct and reproducible method, compare against a credible benchmark, provide honest evaluation, and communicate the result through a useful application.
