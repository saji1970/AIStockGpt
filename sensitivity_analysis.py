import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class SensitivityAnalysis:
    """
    Comprehensive sensitivity analysis for LSTM stock prediction models.
    """
    
    def __init__(self, model, scaler_X=None, scaler_y=None, feature_names=None):
        """
        Initialize the sensitivity analysis.
        
        Args:
            model: Trained LSTM model
            scaler_X: Feature scaler
            scaler_y: Target scaler
            feature_names (list): List of feature names
        """
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.feature_names = feature_names
        self.analysis_results = {}
        
    def feature_sensitivity_analysis(self, X_sample, y_true=None, method='perturbation'):
        """
        Perform feature sensitivity analysis.
        
        Args:
            X_sample (np.array): Sample data for analysis
            y_true (np.array): True values for comparison
            method (str): Analysis method ('perturbation', 'correlation', 'gradient')
            
        Returns:
            dict: Sensitivity analysis results
        """
        if method == 'perturbation':
            return self._perturbation_analysis(X_sample, y_true)
        elif method == 'correlation':
            return self._correlation_analysis(X_sample, y_true)
        elif method == 'gradient':
            return self._gradient_analysis(X_sample)
        else:
            raise ValueError("Method must be 'perturbation', 'correlation', or 'gradient'")
    
    def _perturbation_analysis(self, X_sample, y_true=None):
        """
        Perform perturbation-based sensitivity analysis.
        
        Args:
            X_sample (np.array): Sample data
            y_true (np.array): True values
            
        Returns:
            dict: Perturbation analysis results
        """
        n_features = X_sample.shape[2]
        sensitivity_scores = {}
        
        # Get baseline prediction
        baseline_pred = self.model.predict(X_sample)
        
        # Test different perturbation levels
        perturbation_levels = [0.1, 0.2, 0.5, 1.0]
        
        for level in perturbation_levels:
            level_scores = []
            
            for i in range(n_features):
                # Create perturbed data
                X_perturbed = X_sample.copy()
                
                # Add noise to the feature
                noise = np.random.normal(0, level, X_perturbed.shape[0])
                X_perturbed[:, :, i] += noise.reshape(-1, 1)
                
                # Get prediction with perturbed feature
                perturbed_pred = self.model.predict(X_perturbed)
                
                # Calculate sensitivity as change in prediction
                sensitivity = np.mean(np.abs(perturbed_pred - baseline_pred))
                level_scores.append(sensitivity)
            
            sensitivity_scores[f'level_{level}'] = level_scores
        
        # Calculate overall sensitivity
        overall_sensitivity = np.mean([scores for scores in sensitivity_scores.values()], axis=0)
        
        results = {
            'overall_sensitivity': overall_sensitivity,
            'perturbation_levels': sensitivity_scores,
            'feature_ranking': self._rank_features(overall_sensitivity)
        }
        
        self.analysis_results['perturbation'] = results
        return results
    
    def _correlation_analysis(self, X_sample, y_true):
        """
        Perform correlation-based sensitivity analysis.
        
        Args:
            X_sample (np.array): Sample data
            y_true (np.array): True values
            
        Returns:
            dict: Correlation analysis results
        """
        n_features = X_sample.shape[2]
        correlation_scores = {}
        
        # Get predictions
        predictions = self.model.predict(X_sample).flatten()
        
        for i in range(n_features):
            feature_values = X_sample[:, -1, i]  # Use last timestep
            
            # Calculate correlations
            pearson_corr, _ = pearsonr(feature_values, predictions)
            spearman_corr, _ = spearmanr(feature_values, predictions)
            
            correlation_scores[f'Feature_{i}'] = {
                'pearson': pearson_corr,
                'spearman': spearman_corr,
                'abs_pearson': abs(pearson_corr),
                'abs_spearman': abs(spearman_corr)
            }
        
        # Calculate overall correlation strength
        overall_correlation = np.mean([scores['abs_pearson'] for scores in correlation_scores.values()])
        
        results = {
            'correlation_scores': correlation_scores,
            'overall_correlation': overall_correlation,
            'feature_ranking': self._rank_features([scores['abs_pearson'] for scores in correlation_scores.values()])
        }
        
        self.analysis_results['correlation'] = results
        return results
    
    def _gradient_analysis(self, X_sample):
        """
        Perform gradient-based sensitivity analysis.
        
        Args:
            X_sample (np.array): Sample data
            
        Returns:
            dict: Gradient analysis results
        """
        # This is a simplified gradient analysis
        # In practice, you might want to use TensorFlow's gradient tape for more accurate gradients
        
        n_features = X_sample.shape[2]
        gradient_scores = []
        
        for i in range(n_features):
            # Create small perturbation
            X_perturbed = X_sample.copy()
            epsilon = 1e-6
            X_perturbed[:, :, i] += epsilon
            
            # Calculate gradient approximation
            pred_original = self.model.predict(X_sample)
            pred_perturbed = self.model.predict(X_perturbed)
            
            gradient = np.mean((pred_perturbed - pred_original) / epsilon)
            gradient_scores.append(abs(gradient))
        
        results = {
            'gradient_scores': gradient_scores,
            'feature_ranking': self._rank_features(gradient_scores)
        }
        
        self.analysis_results['gradient'] = results
        return results
    
    def _rank_features(self, scores):
        """
        Rank features based on sensitivity scores.
        
        Args:
            scores (list): Sensitivity scores
            
        Returns:
            list: Ranked feature indices
        """
        return np.argsort(scores)[::-1]
    
    def parameter_sensitivity_analysis(self, X_train, y_train, param_ranges):
        """
        Perform parameter sensitivity analysis.
        
        Args:
            X_train (np.array): Training data
            y_train (np.array): Training targets
            param_ranges (dict): Parameter ranges to test
            
        Returns:
            dict: Parameter sensitivity results
        """
        param_results = {}
        
        for param_name, param_values in param_ranges.items():
            param_scores = []
            
            for value in param_values:
                # Create new model with different parameter
                if param_name == 'lstm_units':
                    model = self._create_model_with_param(param_name, value)
                elif param_name == 'dropout_rate':
                    model = self._create_model_with_param(param_name, value)
                elif param_name == 'learning_rate':
                    model = self._create_model_with_param(param_name, value)
                
                # Train and evaluate
                history = model.fit(X_train, y_train, epochs=10, verbose=0)
                val_loss = min(history.history['val_loss'])
                param_scores.append(val_loss)
            
            param_results[param_name] = {
                'values': param_values,
                'scores': param_scores,
                'best_value': param_values[np.argmin(param_scores)]
            }
        
        self.analysis_results['parameter'] = param_results
        return param_results
    
    def _create_model_with_param(self, param_name, value):
        """
        Create a model with a specific parameter value.
        
        Args:
            param_name (str): Parameter name
            value: Parameter value
            
        Returns:
            model: Model with the specified parameter
        """
        # This is a simplified version - you would need to implement based on your model architecture
        from lstm_model import LSTMModel
        
        if param_name == 'lstm_units':
            model = LSTMModel(sequence_length=self.model.sequence_length, 
                            n_features=self.model.n_features)
            model.build_model(lstm_units=value)
        elif param_name == 'dropout_rate':
            model = LSTMModel(sequence_length=self.model.sequence_length, 
                            n_features=self.model.n_features)
            model.build_model(dropout_rate=value)
        elif param_name == 'learning_rate':
            model = LSTMModel(sequence_length=self.model.sequence_length, 
                            n_features=self.model.n_features)
            model.build_model(learning_rate=value)
        
        return model
    
    def market_condition_sensitivity(self, X_sample, market_scenarios):
        """
        Analyze model sensitivity to different market conditions.
        
        Args:
            X_sample (np.array): Sample data
            market_scenarios (dict): Different market scenarios
            
        Returns:
            dict: Market condition sensitivity results
        """
        scenario_results = {}
        
        baseline_pred = self.model.predict(X_sample)
        
        for scenario_name, scenario_params in market_scenarios.items():
            X_scenario = X_sample.copy()
            
            # Apply scenario modifications
            for feature_idx, modification in scenario_params.items():
                if modification['type'] == 'multiply':
                    X_scenario[:, :, feature_idx] *= modification['value']
                elif modification['type'] == 'add':
                    X_scenario[:, :, feature_idx] += modification['value']
                elif modification['type'] == 'noise':
                    noise = np.random.normal(0, modification['value'], X_scenario.shape[:2])
                    X_scenario[:, :, feature_idx] += noise
            
            # Get predictions for scenario
            scenario_pred = self.model.predict(X_scenario)
            
            # Calculate impact
            impact = np.mean(np.abs(scenario_pred - baseline_pred))
            
            scenario_results[scenario_name] = {
                'predictions': scenario_pred,
                'impact': impact,
                'percentage_change': np.mean((scenario_pred - baseline_pred) / baseline_pred) * 100
            }
        
        self.analysis_results['market_conditions'] = scenario_results
        return scenario_results
    
    def plot_sensitivity_results(self, analysis_type='perturbation', top_n=10):
        """
        Plot sensitivity analysis results.
        
        Args:
            analysis_type (str): Type of analysis to plot
            top_n (int): Number of top features to show
        """
        if analysis_type not in self.analysis_results:
            print(f"No {analysis_type} analysis results available")
            return
        
        results = self.analysis_results[analysis_type]
        
        if analysis_type == 'perturbation':
            self._plot_perturbation_results(results, top_n)
        elif analysis_type == 'correlation':
            self._plot_correlation_results(results, top_n)
        elif analysis_type == 'parameter':
            self._plot_parameter_results(results)
        elif analysis_type == 'market_conditions':
            self._plot_market_condition_results(results)
    
    def _plot_perturbation_results(self, results, top_n):
        """
        Plot perturbation analysis results.
        
        Args:
            results (dict): Perturbation analysis results
            top_n (int): Number of top features to show
        """
        overall_sensitivity = results['overall_sensitivity']
        feature_ranking = results['feature_ranking'][:top_n]
        
        # Create feature names
        if self.feature_names:
            feature_names = [self.feature_names[i] for i in feature_ranking]
        else:
            feature_names = [f'Feature_{i}' for i in feature_ranking]
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Overall sensitivity
        plt.subplot(1, 2, 1)
        plt.barh(range(len(feature_names)), 
                [overall_sensitivity[i] for i in feature_ranking])
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Sensitivity Score')
        plt.title('Feature Sensitivity (Overall)')
        plt.gca().invert_yaxis()
        
        # Perturbation levels
        plt.subplot(1, 2, 2)
        perturbation_levels = results['perturbation_levels']
        levels = list(perturbation_levels.keys())
        
        for i, level in enumerate(levels):
            level_scores = [perturbation_levels[level][j] for j in feature_ranking]
            plt.plot(range(len(feature_names)), level_scores, 
                    marker='o', label=f'Level {level}')
        
        plt.xticks(range(len(feature_names)), feature_names, rotation=45)
        plt.ylabel('Sensitivity Score')
        plt.title('Sensitivity by Perturbation Level')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def _plot_correlation_results(self, results, top_n):
        """
        Plot correlation analysis results.
        
        Args:
            results (dict): Correlation analysis results
            top_n (int): Number of top features to show
        """
        correlation_scores = results['correlation_scores']
        
        # Sort by absolute Pearson correlation
        sorted_features = sorted(correlation_scores.items(), 
                               key=lambda x: x[1]['abs_pearson'], reverse=True)[:top_n]
        
        feature_names = [item[0] for item in sorted_features]
        pearson_scores = [item[1]['pearson'] for item in sorted_features]
        spearman_scores = [item[1]['spearman'] for item in sorted_features]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pearson correlation
        ax1.barh(range(len(feature_names)), pearson_scores)
        ax1.set_yticks(range(len(feature_names)))
        ax1.set_yticklabels(feature_names)
        ax1.set_xlabel('Pearson Correlation')
        ax1.set_title('Pearson Correlation with Predictions')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.invert_yaxis()
        
        # Spearman correlation
        ax2.barh(range(len(feature_names)), spearman_scores)
        ax2.set_yticks(range(len(feature_names)))
        ax2.set_yticklabels(feature_names)
        ax2.set_xlabel('Spearman Correlation')
        ax2.set_title('Spearman Correlation with Predictions')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    def _plot_parameter_results(self, results):
        """
        Plot parameter sensitivity results.
        
        Args:
            results (dict): Parameter sensitivity results
        """
        n_params = len(results)
        fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
        
        if n_params == 1:
            axes = [axes]
        
        for i, (param_name, param_data) in enumerate(results.items()):
            axes[i].plot(param_data['values'], param_data['scores'], 'o-')
            axes[i].axvline(x=param_data['best_value'], color='red', 
                           linestyle='--', label=f'Best: {param_data["best_value"]}')
            axes[i].set_xlabel(param_name)
            axes[i].set_ylabel('Validation Loss')
            axes[i].set_title(f'{param_name} Sensitivity')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_market_condition_results(self, results):
        """
        Plot market condition sensitivity results.
        
        Args:
            results (dict): Market condition sensitivity results
        """
        scenario_names = list(results.keys())
        impacts = [results[name]['impact'] for name in scenario_names]
        percentage_changes = [results[name]['percentage_change'] for name in scenario_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Impact scores
        ax1.bar(scenario_names, impacts)
        ax1.set_xlabel('Market Scenario')
        ax1.set_ylabel('Impact Score')
        ax1.set_title('Model Sensitivity to Market Conditions')
        ax1.tick_params(axis='x', rotation=45)
        
        # Percentage changes
        ax2.bar(scenario_names, percentage_changes)
        ax2.set_xlabel('Market Scenario')
        ax2.set_ylabel('Percentage Change (%)')
        ax2.set_title('Prediction Change by Scenario')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_sensitivity_report(self, output_file='sensitivity_report.html'):
        """
        Generate a comprehensive HTML report of all sensitivity analyses.
        
        Args:
            output_file (str): Output file path
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>LSTM Stock Prediction - Sensitivity Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>LSTM Stock Prediction - Sensitivity Analysis Report</h1>
        """
        
        # Add each analysis section
        for analysis_type, results in self.analysis_results.items():
            html_content += f"<div class='section'><h2>{analysis_type.title()} Analysis</h2>"
            
            if analysis_type == 'perturbation':
                html_content += self._generate_perturbation_html(results)
            elif analysis_type == 'correlation':
                html_content += self._generate_correlation_html(results)
            elif analysis_type == 'parameter':
                html_content += self._generate_parameter_html(results)
            elif analysis_type == 'market_conditions':
                html_content += self._generate_market_condition_html(results)
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Sensitivity report saved to {output_file}")
    
    def _generate_perturbation_html(self, results):
        """Generate HTML for perturbation analysis results."""
        html = "<h3>Feature Sensitivity Rankings</h3>"
        html += "<table><tr><th>Rank</th><th>Feature</th><th>Sensitivity Score</th></tr>"
        
        overall_sensitivity = results['overall_sensitivity']
        feature_ranking = results['feature_ranking']
        
        for i, feature_idx in enumerate(feature_ranking[:10]):
            feature_name = self.feature_names[feature_idx] if self.feature_names else f"Feature_{feature_idx}"
            score = overall_sensitivity[feature_idx]
            html += f"<tr><td>{i+1}</td><td>{feature_name}</td><td>{score:.4f}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_correlation_html(self, results):
        """Generate HTML for correlation analysis results."""
        html = "<h3>Feature Correlation Rankings</h3>"
        html += "<table><tr><th>Rank</th><th>Feature</th><th>Pearson</th><th>Spearman</th></tr>"
        
        correlation_scores = results['correlation_scores']
        sorted_features = sorted(correlation_scores.items(), 
                               key=lambda x: x[1]['abs_pearson'], reverse=True)
        
        for i, (feature_name, scores) in enumerate(sorted_features[:10]):
            html += f"<tr><td>{i+1}</td><td>{feature_name}</td><td>{scores['pearson']:.4f}</td><td>{scores['spearman']:.4f}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_parameter_html(self, results):
        """Generate HTML for parameter analysis results."""
        html = "<h3>Parameter Sensitivity Results</h3>"
        
        for param_name, param_data in results.items():
            html += f"<h4>{param_name}</h4>"
            html += f"<p>Best value: {param_data['best_value']}</p>"
            html += "<table><tr><th>Value</th><th>Validation Loss</th></tr>"
            
            for value, score in zip(param_data['values'], param_data['scores']):
                html += f"<tr><td>{value}</td><td>{score:.4f}</td></tr>"
            
            html += "</table>"
        
        return html
    
    def _generate_market_condition_html(self, results):
        """Generate HTML for market condition analysis results."""
        html = "<h3>Market Condition Sensitivity</h3>"
        html += "<table><tr><th>Scenario</th><th>Impact Score</th><th>Percentage Change</th></tr>"
        
        for scenario_name, scenario_data in results.items():
            html += f"<tr><td>{scenario_name}</td><td>{scenario_data['impact']:.4f}</td><td>{scenario_data['percentage_change']:.2f}%</td></tr>"
        
        html += "</table>"
        return html

if __name__ == "__main__":
    print("Sensitivity Analysis class created successfully!")
    print("Use this class to analyze the sensitivity of your LSTM stock prediction model.")
