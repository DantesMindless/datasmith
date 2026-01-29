export type ConfigField = {
  key: string;
  label: string;
  type: "number" | "text" | "select" | "layers";
  default?: any;
  helperText?: string;
  options?: { label: string; value: any }[];
};

export const modelConfigSchemas: Record<string, ConfigField[]> = {
  // ============ TRADITIONAL ML ============

  logistic_regression: [
    { key: "max_iter", label: "Max Iterations", type: "number", default: 1000, helperText: "Maximum iterations for convergence (increase if not converging)" },
    {
      key: "solver", label: "Solver", type: "select", default: "saga", helperText: "Optimization algorithm (saga supports all penalties)",
      options: [
        { label: "saga (recommended - supports all)", value: "saga" },
        { label: "lbfgs (L2 only, good for small data)", value: "lbfgs" },
        { label: "liblinear (L1/L2 only, binary)", value: "liblinear" },
        { label: "newton-cg (L2 only)", value: "newton-cg" },
        { label: "sag (L2 only, large data)", value: "sag" },
      ],
    },
    {
      key: "penalty", label: "Penalty", type: "select", default: "l2", helperText: "Regularization (L1: saga/liblinear, ElasticNet: saga only)",
      options: [
        { label: "L2 (Ridge) - all solvers", value: "l2" },
        { label: "L1 (Lasso) - saga/liblinear", value: "l1" },
        { label: "ElasticNet - saga only", value: "elasticnet" },
        { label: "None - saga/lbfgs/newton-cg", value: "none" },
      ],
    },
    { key: "C", label: "Regularization (C)", type: "number", default: 1.0, helperText: "Inverse of regularization strength (smaller = stronger)" },
    {
      key: "class_weight", label: "Class Weight", type: "select", default: "balanced", helperText: "Handle imbalanced classes",
      options: [
        { label: "Balanced (recommended)", value: "balanced" },
        { label: "None", value: null },
      ],
    },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing (0.1-0.3)" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
    { key: "tolerance", label: "Tolerance", type: "number", default: 0.0001, helperText: "Tolerance for stopping criteria" },
    { key: "fit_intercept", label: "Fit Intercept", type: "select", default: true, helperText: "Whether to calculate intercept",
      options: [
        { label: "Yes (recommended)", value: true },
        { label: "No", value: false },
      ],
    },
    {
      key: "multi_class", label: "Multi-class", type: "select", default: "auto", helperText: "Strategy (multinomial not supported by liblinear)",
      options: [
        { label: "Auto (recommended)", value: "auto" },
        { label: "OvR (One vs Rest)", value: "ovr" },
        { label: "Multinomial (not liblinear)", value: "multinomial" },
      ],
    },
    { key: "l1_ratio", label: "L1 Ratio", type: "number", default: 0.5, helperText: "ElasticNet only: mixing (0=L2, 1=L1)" },
    { key: "n_jobs", label: "Parallel Jobs", type: "number", default: -1, helperText: "CPU cores (-1 = all cores)" },
  ],

  linear_regression: [
    {
      key: "fit_intercept", label: "Fit Intercept", type: "select", default: true, helperText: "Whether to calculate the intercept (y-intercept)",
      options: [
        { label: "Yes (recommended)", value: true },
        { label: "No (force through origin)", value: false },
      ],
    },
    {
      key: "copy_X", label: "Copy X", type: "select", default: true, helperText: "Whether to copy X data (set False for large datasets)",
      options: [
        { label: "Yes (recommended)", value: true },
        { label: "No (modify in-place)", value: false },
      ],
    },
    {
      key: "positive", label: "Positive Coefficients", type: "select", default: false, helperText: "Force coefficients to be positive",
      options: [
        { label: "No (allow negative)", value: false },
        { label: "Yes (force positive)", value: true },
      ],
    },
    { key: "n_jobs", label: "Parallel Jobs", type: "number", default: -1, helperText: "CPU cores for computation (-1 = all cores)" },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing (0.1-0.3)" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  decision_tree: [
    { key: "max_depth", label: "Max Depth", type: "number", default: 10, helperText: "Maximum depth of the tree (prevents overfitting, None = unlimited)" },
    { key: "min_samples_split", label: "Min Samples Split", type: "number", default: 5, helperText: "Minimum samples to split a node (higher = less overfitting)" },
    { key: "min_samples_leaf", label: "Min Samples Leaf", type: "number", default: 2, helperText: "Minimum samples in a leaf node (higher = simpler tree)" },
    {
      key: "criterion", label: "Criterion", type: "select", default: "gini", helperText: "Split quality metric (Classification: gini/entropy, Regression: auto-selects squared_error)",
      options: [
        { label: "Gini (classification)", value: "gini" },
        { label: "Entropy (classification)", value: "entropy" },
        { label: "Squared Error (regression)", value: "squared_error" },
        { label: "Absolute Error (regression)", value: "absolute_error" },
      ],
    },
    {
      key: "splitter", label: "Splitter", type: "select", default: "best", helperText: "Strategy to choose split at each node",
      options: [
        { label: "Best", value: "best" },
        { label: "Random", value: "random" },
      ],
    },
    { key: "max_features", label: "Max Features", type: "text", default: "sqrt", helperText: "Features to consider for split ('sqrt', 'log2', or int)" },
    {
      key: "class_weight", label: "Class Weight", type: "select", default: "balanced", helperText: "Handle imbalanced classes (ignored for regression)",
      options: [
        { label: "Balanced (recommended)", value: "balanced" },
        { label: "None", value: null },
      ],
    },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing (0.1-0.3)" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  random_forest: [
    { key: "n_estimators", label: "Number of Trees", type: "number", default: 100, helperText: "Number of trees in the forest (100-500 typical)" },
    { key: "max_depth", label: "Max Depth", type: "number", default: 15, helperText: "Maximum depth of each tree (10-20 prevents overfitting)" },
    { key: "min_samples_split", label: "Min Samples Split", type: "number", default: 5, helperText: "Minimum samples to split a node (higher = less overfitting)" },
    { key: "min_samples_leaf", label: "Min Samples Leaf", type: "number", default: 2, helperText: "Minimum samples in a leaf node (higher = simpler trees)" },
    {
      key: "criterion", label: "Criterion", type: "select", default: "gini", helperText: "Split quality metric (Classification: gini/entropy, Regression: auto-selects squared_error)",
      options: [
        { label: "Gini (classification)", value: "gini" },
        { label: "Entropy (classification)", value: "entropy" },
        { label: "Squared Error (regression)", value: "squared_error" },
        { label: "Absolute Error (regression)", value: "absolute_error" },
      ],
    },
    { key: "max_features", label: "Max Features", type: "text", default: "sqrt", helperText: "Features per tree ('sqrt' recommended, 'log2', or int)" },
    {
      key: "bootstrap", label: "Bootstrap", type: "select", default: true, helperText: "Whether to use bootstrap samples (required for OOB score)",
      options: [
        { label: "True (recommended)", value: true },
        { label: "False", value: false },
      ],
    },
    {
      key: "oob_score", label: "OOB Score", type: "select", default: true, helperText: "Use out-of-bag samples for validation (requires bootstrap=True)",
      options: [
        { label: "True (recommended)", value: true },
        { label: "False", value: false },
      ],
    },
    {
      key: "class_weight", label: "Class Weight", type: "select", default: "balanced", helperText: "Handle imbalanced classes (ignored for regression)",
      options: [
        { label: "Balanced (recommended)", value: "balanced" },
        { label: "Balanced Subsample", value: "balanced_subsample" },
        { label: "None", value: null },
      ],
    },
    { key: "n_jobs", label: "Parallel Jobs", type: "number", default: -1, helperText: "Number of parallel jobs (-1 = all cores)" },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing (0.1-0.3)" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  svm: [
    {
      key: "kernel", label: "Kernel", type: "select", default: "rbf", helperText: "Kernel type (RBF works for most cases)",
      options: [
        { label: "RBF (Gaussian) - recommended", value: "rbf" },
        { label: "Linear (faster, large data)", value: "linear" },
        { label: "Polynomial", value: "poly" },
        { label: "Sigmoid", value: "sigmoid" },
      ],
    },
    { key: "C", label: "Regularization (C)", type: "number", default: 1.0, helperText: "Regularization (0.1-10, smaller = stronger)" },
    { key: "gamma", label: "Gamma", type: "text", default: "scale", helperText: "Kernel coefficient ('scale' recommended, 'auto', or float)" },
    { key: "degree", label: "Degree", type: "number", default: 3, helperText: "Polynomial kernel degree only" },
    {
      key: "class_weight", label: "Class Weight", type: "select", default: "balanced", helperText: "Handle imbalanced classes",
      options: [
        { label: "Balanced (recommended)", value: "balanced" },
        { label: "None", value: null },
      ],
    },
    {
      key: "probability", label: "Probability", type: "select", default: true, helperText: "Enable probability estimates (slower but useful)",
      options: [
        { label: "True (recommended)", value: true },
        { label: "False (faster)", value: false },
      ],
    },
    { key: "max_iter", label: "Max Iterations", type: "number", default: -1, helperText: "Max iterations (-1 = no limit)" },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing (0.1-0.3)" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  naive_bayes: [
    {
      key: "nb_type", label: "Naive Bayes Type", type: "select", default: "gaussian", helperText: "Type (Gaussian for continuous data)",
      options: [
        { label: "GaussianNB (continuous data)", value: "gaussian" },
        { label: "MultinomialNB (count/frequency data)", value: "multinomial" },
        { label: "BernoulliNB (binary features)", value: "bernoulli" },
      ],
    },
    { key: "var_smoothing", label: "Var Smoothing", type: "number", default: 1e-9, helperText: "GaussianNB: variance smoothing (increase if errors)" },
    { key: "alpha", label: "Alpha", type: "number", default: 1.0, helperText: "Multinomial/Bernoulli: smoothing (1.0 = Laplace)" },
    {
      key: "fit_prior", label: "Fit Prior", type: "select", default: true, helperText: "Learn class prior probabilities",
      options: [
        { label: "True (recommended)", value: true },
        { label: "False (uniform prior)", value: false },
      ],
    },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing (0.1-0.3)" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  knn: [
    { key: "n_neighbors", label: "Number of Neighbors (k)", type: "number", default: 5, helperText: "Number of neighbors (3-15 typical, odd for classification)" },
    {
      key: "weights", label: "Weights", type: "select", default: "distance", helperText: "Weight function for predictions",
      options: [
        { label: "Distance (recommended)", value: "distance" },
        { label: "Uniform", value: "uniform" },
      ],
    },
    {
      key: "algorithm", label: "Algorithm", type: "select", default: "auto", helperText: "Algorithm for nearest neighbor search",
      options: [
        { label: "Auto (recommended)", value: "auto" },
        { label: "Ball Tree", value: "ball_tree" },
        { label: "KD Tree", value: "kd_tree" },
        { label: "Brute Force (small data)", value: "brute" },
      ],
    },
    {
      key: "metric", label: "Distance Metric", type: "select", default: "minkowski", helperText: "Distance metric for neighbors",
      options: [
        { label: "Minkowski (p=2 is Euclidean)", value: "minkowski" },
        { label: "Euclidean", value: "euclidean" },
        { label: "Manhattan", value: "manhattan" },
      ],
    },
    { key: "p", label: "Power (p)", type: "number", default: 2, helperText: "Minkowski power (1=Manhattan, 2=Euclidean)" },
    { key: "leaf_size", label: "Leaf Size", type: "number", default: 30, helperText: "Leaf size for Ball/KD Tree (20-50)" },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing (0.1-0.3)" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  // ============ ENSEMBLE METHODS ============

  gradient_boosting: [
    { key: "n_estimators", label: "Number of Estimators", type: "number", default: 100, helperText: "Number of boosting stages (50-300 recommended, max 500)" },
    { key: "learning_rate", label: "Learning Rate", type: "number", default: 0.1, helperText: "Shrinks contribution of each tree (0.05-0.2 typical)" },
    { key: "max_depth", label: "Max Depth", type: "number", default: 3, helperText: "Maximum depth of each tree (3-5 typical, prevents overfitting)" },
    { key: "min_samples_split", label: "Min Samples Split", type: "number", default: 20, helperText: "Minimum samples to split a node (higher prevents overfitting)" },
    { key: "min_samples_leaf", label: "Min Samples Leaf", type: "number", default: 10, helperText: "Minimum samples in a leaf node (higher prevents overfitting)" },
    { key: "subsample", label: "Subsample", type: "number", default: 0.8, helperText: "Fraction of samples per tree (0.7-0.9 reduces overfitting)" },
    { key: "max_features", label: "Max Features", type: "text", default: "sqrt", helperText: "Features per tree ('sqrt' recommended, 'log2', int, or float)" },
    {
      key: "loss", label: "Loss Function", type: "select", default: "log_loss", helperText: "Loss function to optimize",
      options: [
        { label: "Log Loss (Deviance) - recommended", value: "log_loss" },
        { label: "Exponential (AdaBoost)", value: "exponential" },
      ],
    },
    { key: "validation_fraction", label: "Validation Fraction", type: "number", default: 0.1, helperText: "Fraction for early stopping validation (0.1-0.2)" },
    { key: "n_iter_no_change", label: "Early Stopping Patience", type: "number", default: 10, helperText: "Stop if no improvement for N iterations (prevents overfitting)" },
    { key: "tol", label: "Tolerance", type: "number", default: 0.0001, helperText: "Tolerance for early stopping (0.0001-0.001)" },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing (0.1-0.3)" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducible results" },
  ],

  xgboost: [
    { key: "n_estimators", label: "Number of Estimators", type: "number", default: 100, helperText: "Number of boosting rounds (100-500 typical)" },
    { key: "learning_rate", label: "Learning Rate (eta)", type: "number", default: 0.1, helperText: "Step size shrinkage (0.01-0.3, lower = more trees needed)" },
    { key: "max_depth", label: "Max Depth", type: "number", default: 6, helperText: "Maximum tree depth (3-10, lower = less overfitting)" },
    { key: "min_child_weight", label: "Min Child Weight", type: "number", default: 1, helperText: "Minimum sum of instance weight in child (higher = conservative)" },
    { key: "subsample", label: "Subsample", type: "number", default: 0.8, helperText: "Fraction of samples per tree (0.7-0.9 reduces overfitting)" },
    { key: "colsample_bytree", label: "Column Sample by Tree", type: "number", default: 0.8, helperText: "Fraction of features per tree (0.7-0.9)" },
    { key: "gamma", label: "Gamma (min_split_loss)", type: "number", default: 0, helperText: "Minimum loss reduction for split (0-5)" },
    { key: "reg_alpha", label: "L1 Regularization (alpha)", type: "number", default: 0, helperText: "L1 regularization on weights" },
    { key: "reg_lambda", label: "L2 Regularization (lambda)", type: "number", default: 1, helperText: "L2 regularization on weights" },
    {
      key: "booster", label: "Booster", type: "select", default: "gbtree", helperText: "Booster type",
      options: [
        { label: "gbtree (recommended)", value: "gbtree" },
        { label: "dart (dropout)", value: "dart" },
        { label: "gblinear (linear)", value: "gblinear" },
      ],
    },
    {
      key: "objective", label: "Objective", type: "select", default: "auto", helperText: "Learning objective (auto-detected based on target)",
      options: [
        { label: "Auto (recommended)", value: "auto" },
        { label: "Binary Logistic (classification)", value: "binary:logistic" },
        { label: "Multi Softprob (multi-class)", value: "multi:softprob" },
        { label: "Squared Error (regression)", value: "reg:squarederror" },
        { label: "Absolute Error (regression)", value: "reg:absoluteerror" },
      ],
    },
    { key: "scale_pos_weight", label: "Scale Pos Weight", type: "number", default: 1, helperText: "Balance positive/negative weights (classification)" },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing (0.1-0.3)" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  lightgbm: [
    { key: "n_estimators", label: "Number of Estimators", type: "number", default: 100, helperText: "Number of boosting iterations (100-500 typical)" },
    { key: "learning_rate", label: "Learning Rate", type: "number", default: 0.1, helperText: "Boosting learning rate (0.01-0.3)" },
    { key: "max_depth", label: "Max Depth", type: "number", default: -1, helperText: "Max tree depth (-1 = no limit, use num_leaves instead)" },
    { key: "num_leaves", label: "Number of Leaves", type: "number", default: 31, helperText: "Max leaves per tree (main parameter, 2^depth rule)" },
    { key: "min_child_samples", label: "Min Child Samples", type: "number", default: 20, helperText: "Min data in a leaf (higher = less overfitting)" },
    { key: "subsample", label: "Subsample (bagging_fraction)", type: "number", default: 0.8, helperText: "Fraction of data per iteration (0.7-0.9)" },
    { key: "colsample_bytree", label: "Feature Fraction", type: "number", default: 0.8, helperText: "Fraction of features per tree (0.7-0.9)" },
    { key: "reg_alpha", label: "L1 Regularization", type: "number", default: 0, helperText: "L1 regularization (0-1)" },
    { key: "reg_lambda", label: "L2 Regularization", type: "number", default: 0, helperText: "L2 regularization (0-1)" },
    {
      key: "boosting_type", label: "Boosting Type", type: "select", default: "gbdt", helperText: "Boosting algorithm",
      options: [
        { label: "GBDT (recommended)", value: "gbdt" },
        { label: "DART (dropout)", value: "dart" },
        { label: "GOSS (faster, large data)", value: "goss" },
      ],
    },
    {
      key: "objective", label: "Objective", type: "select", default: "auto", helperText: "Learning objective (auto-detected based on target)",
      options: [
        { label: "Auto (recommended)", value: "auto" },
        { label: "Binary (classification)", value: "binary" },
        { label: "Multiclass (classification)", value: "multiclass" },
        { label: "Regression", value: "regression" },
        { label: "Regression MAE", value: "regression_l1" },
      ],
    },
    {
      key: "class_weight", label: "Class Weight", type: "select", default: "balanced", helperText: "Handle imbalanced classes (classification only)",
      options: [
        { label: "Balanced (recommended)", value: "balanced" },
        { label: "None", value: null },
      ],
    },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing (0.1-0.3)" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  adaboost: [
    { key: "n_estimators", label: "Number of Estimators", type: "number", default: 100, helperText: "Number of weak learners (50-200 typical)" },
    { key: "learning_rate", label: "Learning Rate", type: "number", default: 0.5, helperText: "Weight per classifier (0.1-1.0, lower = more estimators needed)" },
    {
      key: "algorithm", label: "Algorithm", type: "select", default: "SAMME.R", helperText: "Boosting algorithm (SAMME.R is faster and more accurate)",
      options: [
        { label: "SAMME.R (recommended)", value: "SAMME.R" },
        { label: "SAMME (discrete)", value: "SAMME" },
      ],
    },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing (0.1-0.3)" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  // ============ REGRESSION MODELS ============

  decision_tree_regressor: [
    { key: "max_depth", label: "Max Depth", type: "number", default: 10, helperText: "Maximum depth of the tree (prevents overfitting)" },
    { key: "min_samples_split", label: "Min Samples Split", type: "number", default: 5, helperText: "Minimum samples to split a node" },
    { key: "min_samples_leaf", label: "Min Samples Leaf", type: "number", default: 2, helperText: "Minimum samples in a leaf node" },
    {
      key: "criterion", label: "Criterion", type: "select", default: "squared_error", helperText: "Function to measure split quality",
      options: [
        { label: "Squared Error (MSE)", value: "squared_error" },
        { label: "Absolute Error (MAE)", value: "absolute_error" },
        { label: "Friedman MSE", value: "friedman_mse" },
        { label: "Poisson", value: "poisson" },
      ],
    },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  random_forest_regressor: [
    { key: "n_estimators", label: "Number of Trees", type: "number", default: 100, helperText: "Number of trees in the forest" },
    { key: "max_depth", label: "Max Depth", type: "number", default: 15, helperText: "Maximum depth of each tree" },
    { key: "min_samples_split", label: "Min Samples Split", type: "number", default: 5, helperText: "Minimum samples to split a node" },
    { key: "min_samples_leaf", label: "Min Samples Leaf", type: "number", default: 2, helperText: "Minimum samples in a leaf node" },
    {
      key: "criterion", label: "Criterion", type: "select", default: "squared_error", helperText: "Function to measure split quality",
      options: [
        { label: "Squared Error (MSE)", value: "squared_error" },
        { label: "Absolute Error (MAE)", value: "absolute_error" },
        { label: "Friedman MSE", value: "friedman_mse" },
        { label: "Poisson", value: "poisson" },
      ],
    },
    { key: "n_jobs", label: "Parallel Jobs", type: "number", default: -1, helperText: "Number of parallel jobs (-1 = all cores)" },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  svr: [
    {
      key: "kernel", label: "Kernel", type: "select", default: "rbf", helperText: "Kernel type",
      options: [
        { label: "RBF (Gaussian)", value: "rbf" },
        { label: "Linear", value: "linear" },
        { label: "Polynomial", value: "poly" },
        { label: "Sigmoid", value: "sigmoid" },
      ],
    },
    { key: "C", label: "Regularization (C)", type: "number", default: 1.0, helperText: "Regularization parameter" },
    { key: "epsilon", label: "Epsilon", type: "number", default: 0.1, helperText: "Epsilon in the epsilon-SVR model" },
    { key: "gamma", label: "Gamma", type: "text", default: "scale", helperText: "Kernel coefficient ('scale', 'auto', or float)" },
    { key: "degree", label: "Degree", type: "number", default: 3, helperText: "Polynomial kernel degree only" },
    { key: "max_iter", label: "Max Iterations", type: "number", default: -1, helperText: "Max iterations (-1 = no limit)" },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  knn_regressor: [
    { key: "n_neighbors", label: "Number of Neighbors (k)", type: "number", default: 5, helperText: "Number of neighbors (3-15 typical)" },
    {
      key: "weights", label: "Weights", type: "select", default: "distance", helperText: "Weight function for predictions",
      options: [
        { label: "Distance (recommended)", value: "distance" },
        { label: "Uniform", value: "uniform" },
      ],
    },
    {
      key: "algorithm", label: "Algorithm", type: "select", default: "auto", helperText: "Algorithm for nearest neighbor search",
      options: [
        { label: "Auto (recommended)", value: "auto" },
        { label: "Ball Tree", value: "ball_tree" },
        { label: "KD Tree", value: "kd_tree" },
        { label: "Brute Force", value: "brute" },
      ],
    },
    { key: "leaf_size", label: "Leaf Size", type: "number", default: 30, helperText: "Leaf size for Ball/KD Tree" },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  gradient_boosting_regressor: [
    { key: "n_estimators", label: "Number of Estimators", type: "number", default: 100, helperText: "Number of boosting stages" },
    { key: "learning_rate", label: "Learning Rate", type: "number", default: 0.1, helperText: "Shrinks contribution of each tree" },
    { key: "max_depth", label: "Max Depth", type: "number", default: 3, helperText: "Maximum depth of each tree" },
    { key: "min_samples_split", label: "Min Samples Split", type: "number", default: 5, helperText: "Minimum samples to split a node" },
    { key: "min_samples_leaf", label: "Min Samples Leaf", type: "number", default: 2, helperText: "Minimum samples in a leaf node" },
    { key: "subsample", label: "Subsample", type: "number", default: 0.8, helperText: "Fraction of samples per tree" },
    {
      key: "loss", label: "Loss Function", type: "select", default: "squared_error", helperText: "Loss function to optimize",
      options: [
        { label: "Squared Error (MSE)", value: "squared_error" },
        { label: "Absolute Error (MAE)", value: "absolute_error" },
        { label: "Huber", value: "huber" },
        { label: "Quantile", value: "quantile" },
      ],
    },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  xgboost_regressor: [
    { key: "n_estimators", label: "Number of Estimators", type: "number", default: 100, helperText: "Number of boosting rounds" },
    { key: "learning_rate", label: "Learning Rate", type: "number", default: 0.1, helperText: "Step size shrinkage" },
    { key: "max_depth", label: "Max Depth", type: "number", default: 6, helperText: "Maximum tree depth" },
    { key: "min_child_weight", label: "Min Child Weight", type: "number", default: 1, helperText: "Minimum sum of instance weight in child" },
    { key: "subsample", label: "Subsample", type: "number", default: 0.8, helperText: "Fraction of samples per tree" },
    { key: "colsample_bytree", label: "Column Sample by Tree", type: "number", default: 0.8, helperText: "Fraction of features per tree" },
    { key: "gamma", label: "Gamma", type: "number", default: 0, helperText: "Minimum loss reduction for split" },
    { key: "reg_alpha", label: "L1 Regularization", type: "number", default: 0, helperText: "L1 regularization on weights" },
    { key: "reg_lambda", label: "L2 Regularization", type: "number", default: 1, helperText: "L2 regularization on weights" },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  lightgbm_regressor: [
    { key: "n_estimators", label: "Number of Estimators", type: "number", default: 100, helperText: "Number of boosting iterations" },
    { key: "learning_rate", label: "Learning Rate", type: "number", default: 0.1, helperText: "Boosting learning rate" },
    { key: "max_depth", label: "Max Depth", type: "number", default: -1, helperText: "Max tree depth (-1 = no limit)" },
    { key: "num_leaves", label: "Number of Leaves", type: "number", default: 31, helperText: "Max leaves per tree" },
    { key: "min_child_samples", label: "Min Child Samples", type: "number", default: 20, helperText: "Min data in a leaf" },
    { key: "subsample", label: "Subsample", type: "number", default: 0.8, helperText: "Fraction of data per iteration" },
    { key: "colsample_bytree", label: "Feature Fraction", type: "number", default: 0.8, helperText: "Fraction of features per tree" },
    { key: "reg_alpha", label: "L1 Regularization", type: "number", default: 0, helperText: "L1 regularization" },
    { key: "reg_lambda", label: "L2 Regularization", type: "number", default: 0, helperText: "L2 regularization" },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  adaboost_regressor: [
    { key: "n_estimators", label: "Number of Estimators", type: "number", default: 100, helperText: "Number of weak learners" },
    { key: "learning_rate", label: "Learning Rate", type: "number", default: 0.5, helperText: "Weight per regressor" },
    {
      key: "loss", label: "Loss Function", type: "select", default: "linear", helperText: "Loss function for weight updates",
      options: [
        { label: "Linear", value: "linear" },
        { label: "Square", value: "square" },
        { label: "Exponential", value: "exponential" },
      ],
    },
    { key: "test_size", label: "Test Size", type: "number", default: 0.2, helperText: "Proportion of data for testing" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  // ============ DEEP LEARNING - GENERAL ============

  neural_network: [
    // Training parameters
    { key: "epochs", label: "Epochs", type: "number", default: 50, helperText: "Number of training epochs (50-200 for tabular data)" },
    { key: "batch_size", label: "Batch Size", type: "number", default: 32, helperText: "Samples per batch (32-128, smaller for small datasets)" },
    { key: "learning_rate", label: "Learning Rate", type: "number", default: 0.001, helperText: "Initial learning rate (0.0001-0.01)" },
    {
      key: "optimizer", label: "Optimizer", type: "select", default: "adam", helperText: "Optimization algorithm",
      options: [
        { label: "Adam (recommended)", value: "adam" },
        { label: "AdamW (with weight decay)", value: "adamw" },
        { label: "SGD (with momentum)", value: "sgd" },
        { label: "RMSprop", value: "rmsprop" },
        { label: "NAdam (Adam with Nesterov)", value: "nadam" },
      ],
    },
    { key: "momentum", label: "Momentum", type: "number", default: 0.9, helperText: "SGD momentum (0.9-0.99, ignored for Adam)" },

    // Architecture
    {
      key: "layer_config",
      label: "Hidden Layers",
      type: "layers",
      helperText: "Configure hidden layers (start wide, narrow down)",
      default: [
        { units: 128, activation: "relu" },
        { units: 64, activation: "relu" },
        { units: 32, activation: "relu" },
      ],
    },
    {
      key: "use_batch_norm", label: "Batch Normalization", type: "select", default: true, helperText: "Normalize layer inputs (improves training)",
      options: [
        { label: "True (recommended)", value: true },
        { label: "False", value: false },
      ],
    },
    {
      key: "init_method", label: "Weight Initialization", type: "select", default: "kaiming", helperText: "How to initialize weights",
      options: [
        { label: "Kaiming/He (recommended for ReLU)", value: "kaiming" },
        { label: "Xavier/Glorot (for tanh/sigmoid)", value: "xavier" },
        { label: "Normal (std=0.01)", value: "normal" },
        { label: "Uniform", value: "uniform" },
      ],
    },

    // Regularization
    { key: "dropout", label: "Dropout Rate", type: "number", default: 0.2, helperText: "Dropout probability (0.1-0.5, prevents overfitting)" },
    { key: "weight_decay", label: "Weight Decay (L2)", type: "number", default: 0.0001, helperText: "L2 regularization strength (0.0001-0.01)" },
    {
      key: "label_smoothing", label: "Label Smoothing", type: "number", default: 0.0, helperText: "Smoothing factor (0.1-0.2 for classification)"
    },

    // Learning rate scheduling
    {
      key: "lr_scheduler", label: "LR Scheduler", type: "select", default: "plateau", helperText: "Learning rate decay strategy",
      options: [
        { label: "None", value: "none" },
        { label: "Reduce on Plateau (recommended)", value: "plateau" },
        { label: "Cosine Annealing", value: "cosine" },
        { label: "Step LR", value: "step" },
        { label: "Exponential Decay", value: "exponential" },
        { label: "One Cycle", value: "one_cycle" },
      ],
    },
    { key: "lr_patience", label: "LR Patience", type: "number", default: 5, helperText: "Epochs before reducing LR (for plateau scheduler)" },
    { key: "lr_factor", label: "LR Reduction Factor", type: "number", default: 0.5, helperText: "Multiply LR by this when reducing (0.1-0.5)" },
    { key: "min_lr", label: "Minimum LR", type: "number", default: 0.000001, helperText: "Stop reducing LR below this value" },

    // Early stopping
    {
      key: "early_stopping", label: "Early Stopping", type: "select", default: true, helperText: "Stop training when validation loss stops improving",
      options: [
        { label: "True (recommended)", value: true },
        { label: "False", value: false },
      ],
    },
    { key: "early_stopping_patience", label: "Early Stop Patience", type: "number", default: 10, helperText: "Epochs to wait before stopping" },
    { key: "early_stopping_min_delta", label: "Min Delta", type: "number", default: 0.0001, helperText: "Minimum change to qualify as improvement" },

    // Loss function
    {
      key: "loss_function", label: "Loss Function", type: "select", default: "cross_entropy", helperText: "Loss function to optimize",
      options: [
        { label: "Cross Entropy (classification)", value: "cross_entropy" },
        { label: "Focal Loss (imbalanced classes)", value: "focal" },
        { label: "MSE (regression)", value: "mse" },
        { label: "MAE / L1 (regression)", value: "mae" },
        { label: "Huber (robust regression)", value: "huber" },
        { label: "NLL Loss", value: "nll" },
      ],
    },
    { key: "focal_gamma", label: "Focal Loss Gamma", type: "number", default: 2.0, helperText: "Focusing parameter for focal loss (1-5)" },
    {
      key: "class_weights", label: "Use Class Weights", type: "select", default: true, helperText: "Balance classes automatically (classification)",
      options: [
        { label: "True (handle imbalance)", value: true },
        { label: "False", value: false },
      ],
    },

    // Advanced options
    { key: "gradient_clipping", label: "Gradient Clipping", type: "number", default: 1.0, helperText: "Max gradient norm (0 = disabled)" },
    { key: "warmup_epochs", label: "Warmup Epochs", type: "number", default: 0, helperText: "Gradually increase LR from 0 (0-5)" },
    {
      key: "normalize_features", label: "Feature Normalization", type: "select", default: "standard", helperText: "How to normalize input features",
      options: [
        { label: "Standard (zero mean, unit var)", value: "standard" },
        { label: "MinMax (0-1 range)", value: "minmax" },
        { label: "Robust (median/IQR)", value: "robust" },
        { label: "None", value: "none" },
      ],
    },

    // Data split
    { key: "test_size", label: "Test/Validation Split", type: "number", default: 0.2, helperText: "Fraction for testing (0.1-0.3)" },
    { key: "random_state", label: "Random State", type: "number", default: 42, helperText: "Random seed for reproducibility" },
  ],

  // ============ DEEP LEARNING - COMPUTER VISION ============

  cnn: [
    { key: "epochs", label: "Epochs", type: "number", default: 50, helperText: "Number of training epochs (50+ for complex datasets)" },
    { key: "learning_rate", label: "Learning Rate", type: "number", default: 0.0001, helperText: "Initial learning rate (lower for many classes)" },
    { key: "batch_size", label: "Batch Size", type: "number", default: 32, helperText: "Number of samples per batch (32 optimal for 100+ classes)" },
    { key: "input_size", label: "Input Size", type: "number", default: 224, helperText: "Image resize dimension (224 recommended for complex classification)" },
    { key: "weight_decay", label: "Weight Decay (L2)", type: "number", default: 0.0001, helperText: "L2 regularization strength" },
    { key: "early_stopping_patience", label: "Early Stopping Patience", type: "number", default: 7, helperText: "Epochs to wait before early stopping" },
    {
      key: "optimizer", label: "Optimizer", type: "select", default: "adamw", helperText: "Optimization algorithm",
      options: [
        { label: "AdamW (best for complex tasks)", value: "adamw" },
        { label: "Adam", value: "adam" },
        { label: "SGD with Momentum", value: "sgd" },
        { label: "RMSprop", value: "rmsprop" },
      ],
    },
    {
      key: "scheduler", label: "Learning Rate Scheduler", type: "select", default: "cosine", helperText: "LR scheduling strategy",
      options: [
        { label: "None", value: "none" },
        { label: "Cosine Annealing (recommended)", value: "cosine" },
        { label: "Step LR", value: "step" },
        { label: "Reduce on Plateau", value: "plateau" },
        { label: "Exponential Decay", value: "exponential" },
      ],
    },
    {
      key: "conv_layers",
      label: "Convolution Layers",
      type: "layers",
      helperText: "Configure conv layers (deeper for complex datasets)",
      default: [
        { out_channels: 64, kernel_size: 3, activation: "relu", dropout: 0.1 },
        { out_channels: 128, kernel_size: 3, activation: "relu", dropout: 0.2 },
        { out_channels: 256, kernel_size: 3, activation: "relu", dropout: 0.3 },
        { out_channels: 512, kernel_size: 3, activation: "relu", dropout: 0.4 },
      ],
    },
    {
      key: "fc_layers",
      label: "Fully Connected Layers",
      type: "layers",
      helperText: "Configure FC layers after conv (larger for 100+ classes)",
      default: [
        { units: 1024, activation: "relu", dropout: 0.5 },
        { units: 512, activation: "relu", dropout: 0.5 },
      ],
    },
    { key: "global_dropout", label: "Global Dropout Rate", type: "number", default: 0.5, helperText: "Final dropout before classification" },
    {
      key: "use_batch_norm", label: "Batch Normalization", type: "select", default: true, helperText: "Use batch normalization (recommended)",
      options: [
        { label: "True (recommended)", value: true },
        { label: "False", value: false },
      ],
    },
    {
      key: "data_augmentation", label: "Data Augmentation", type: "select", default: "advanced", helperText: "Augmentation strategy",
      options: [
        { label: "Advanced (rotation, flip, zoom, brightness)", value: "advanced" },
        { label: "Basic (flip, rotation)", value: "basic" },
        { label: "None", value: "none" },
      ],
    },
    {
      key: "augmentation_strength", label: "Augmentation Strength", type: "select", default: "medium", helperText: "How aggressive augmentation should be",
      options: [
        { label: "Light (10-15% variation)", value: "light" },
        { label: "Medium (20-25% variation)", value: "medium" },
        { label: "Strong (30-35% variation)", value: "strong" },
      ],
    },
    {
      key: "label_smoothing", label: "Label Smoothing", type: "number", default: 0.1, helperText: "Smoothing factor (0.1-0.2 for many classes)" 
    },
    {
      key: "gradient_clipping", label: "Gradient Clipping", type: "number", default: 1.0, helperText: "Max gradient norm (prevents exploding gradients)" 
    },
    {
      key: "class_weights", label: "Use Class Weights", type: "select", default: true, helperText: "Balance classes automatically",
      options: [
        { label: "True (handle imbalance)", value: true },
        { label: "False", value: false },
      ],
    },
    { key: "mixup_alpha", label: "MixUp Alpha", type: "number", default: 0.2, helperText: "MixUp augmentation strength (0 = disabled)" },
    { key: "test_size", label: "Validation Split", type: "number", default: 0.15, helperText: "Fraction for validation (0.15 = 15%)" },
  ],
};
