/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   classifier.cpp                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/03/23 12:47:38 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/03/23 12:47:38 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Linear.hpp"


Row sigmoid(const Row& z)
{
    Row result(z.size());
    for (size_t i = 0; i < z.size(); ++i)
        result[i] = 1.0 / (1.0 + std::exp(-z[i]));
    return result;
}


double crossEntropyCost(const Row& h, const Row& y)
{
    double cost = 0.0;
    size_t m = y.size();
    
    for (size_t i = 0; i < m; ++i) {
        double h_val = std::max(1e-15, std::min(1.0 - 1e-15, h[i]));        
        cost -= y[i] * std::log(h_val) + (1.0 - y[i]) * std::log(1.0 - h_val);
    }
    return cost / m;
}


Row logisticRegression(const Matrix& X, const Row& y)
{
    size_t m = X.size();
    size_t n = X[0].size();
    
    Row w(n, 0.0);
    Row cost_history;
    double prev_cost = 1e9;
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++)
    {
        Row h = sigmoid(X * w);
        double cost = crossEntropyCost(h, y);
        cost_history.push_back(cost);

        if (epoch % 1000 == 0)
            std::cout << "Epoch " << epoch << " | Cost: " << cost << "\r" << std::flush;

        if (std::abs(prev_cost - cost) < TOLERANCE)
        {
            std::cout << "Epoch " << epoch  << " | Final Cost: " << cost << "\n";
            break;
        }
        prev_cost = cost;

        Row error(m);
        for (size_t i = 0; i < m; ++i)
            error[i] = h[i] - y[i];

        Row gradient = transpose(X) * error; 
        for (size_t j = 0; j < n; ++j)
            w[j] -= (LR / m) * gradient[j];
    }
    appendCSV("cost_history.csv", "sigm cost", cost_history);
    return w;
}

Row gda(const Matrix& X, const Row& y)
{
    size_t m = X.size();
    size_t n_cols = X[0].size();
    size_t n_features = n_cols - 1;

    double m0 = 0.0, m1 = 0.0;
    Row mu0(n_features, 0.0);
    Row mu1(n_features, 0.0);

    for (size_t i = 0; i < m; i++) 
    {
        if (y[i] == 1.0 && ++m1)
            for (size_t j = 1; j < n_cols; j++)
                mu1[j - 1] += X[i][j];
        else if (++m0)
            for (size_t j = 1; j < n_cols; j++)
                mu0[j - 1] += X[i][j];
    }
    for (size_t j = 0; j < n_features; j++)
    {
        mu0[j] /= m0;
        mu1[j] /= m1;
    }

    double phi = m1 / m;
    Matrix Sigma(n_features, Row(n_features, 0.0));
    for (size_t i = 0; i < m; ++i) {
        Row res(n_features);
        for (size_t j = 1; j < n_cols; ++j)
            res[j - 1] = X[i][j] - (y[i] == 1.0 ? mu1[j - 1] : mu0[j - 1]);
        for (size_t r = 0; r < n_features; ++r)
            for (size_t c = 0; c < n_features; ++c)
                Sigma[r][c] += res[r] * res[c];
    }
    for (size_t r = 0; r < n_features; ++r)
        for (size_t c = 0; c < n_features; ++c)
            Sigma[r][c] /= m;

    Matrix SigmaInv = invert(Sigma);

    // w = SigmaInv * (mu1 - mu0)
    Row w(n_features+1, 0.0);
    for (size_t r = 0; r < n_features; ++r)
        for (size_t c = 0; c < n_features; ++c)
            w[r+1] += SigmaInv[r][c] * (mu1[c] - mu0[c]);

    // w0 = -0.5 * mu1^T * SigmaInv * mu1 + 0.5 * mu0^T * SigmaInv * mu0 + ln(phi / (1-phi))
    double term1 = 0, term2 = 0;
    for (size_t r = 0; r < n_features; ++r) {
        double temp1 = 0, temp2 = 0;
        for (size_t c = 0; c < n_features; ++c) {
            temp1 += SigmaInv[r][c] * mu1[c];
            temp2 += SigmaInv[r][c] * mu0[c];
        }
        term1 += mu1[r] * temp1;
        term2 += mu0[r] * temp2;
    }
    w[0] = -0.5 * term1 + 0.5 * term2 + std::log(phi / (1.0 - phi));
    return w;
}


Row& denormalize(Row& output, double mean, double std)
{
    for (double& x : output)
        x = x * std + mean;
    return output;
}


void exportResults(const Matrix& X, const Row& y_true, const Row& y_pred, const Row& w)
{
    std::string pred_filename = "predictions.csv";
    std::ofstream pred_file(pred_filename);
    
    if (!pred_file.is_open()) {
        std::cerr << "Error: Could not open " << pred_filename << " for writing.\n";
        return;
    }

    pred_file << "x1,x2,actual,predicted\n";
    for (size_t i = 0; i < X.size(); ++i) 
    {
        for (size_t j = 1; j < X[i].size(); ++j)
            pred_file << X[i][j] << ",";
        pred_file << y_true[i] << "," << y_pred[i] << "\n";
    }
    pred_file.close();
    std::cout << "Saved predictions to: " << pred_filename << "\n";

    std::string weight_filename = "weights.csv";
    std::ofstream weight_file(weight_filename);
    
    if (weight_file.is_open()) {
        weight_file << "weight_value\n";
        for (double val : w) {
            weight_file << val << "\n";
        }
        weight_file.close();
        std::cout << "Saved weights to: " << weight_filename << "\n";
    }
}


void doLogReg(t_csv& csv, t_csv& test)
{
    double mean = csv.means[csv.means.size()-1];
    double std = csv.stds[csv.stds.size()-1];
    Row& train_labels = denormalize(csv.output, mean, std);
    Row& test_labels = denormalize(test.output, mean, std);

    Row w = logisticRegression(csv.data, train_labels);
    Row preds = sigmoid(test.data * w);
    double cost = crossEntropyCost(preds, test_labels);
    std::cout << "Test Set Cross-Entropy Cost: " << cost << "\n";

    int correct_count = 0;
    size_t n_test = test_labels.size();
    Row pred_labels(n_test);
    for (size_t i = 0; i < n_test; ++i)
    {
        pred_labels[i] = (preds[i] >= 0.5) ? 1.0 : 0.0;
        if (pred_labels[i] == test_labels[i])
            correct_count++;
    }

    double accuracy = (double)correct_count / n_test * 100.0;
    std::cout << "Test Set Accuracy: " << accuracy << "%\n\n";

    exportResults(test.data, test.output, pred_labels, w);
}


void doGDA(t_csv& csv)
{
    t_csv test = splitTest(csv);
    normalize(csv); 
    normalize(test);
    double mean = csv.means[csv.means.size()-1];
    double std = csv.stds[csv.stds.size()-1];
    Row& train_labels = denormalize(csv.output, mean, std);
    Row& test_labels = denormalize(test.output, mean, std);

    for (size_t i = 0; i < csv.data.size(); ++i)
        csv.data[i].insert(csv.data[i].begin(), 1.0);
    for (size_t i = 0; i < test.data.size(); ++i)
        test.data[i].insert(test.data[i].begin(), 1.0);

    Row w = gda(csv.data, train_labels);
    Row preds = sigmoid(test.data * w);
    double cost = crossEntropyCost(preds, test_labels);
    std::cout << "Test Set Cross-Entropy Cost: " << cost << "\n";

    int correct_count = 0;
    size_t n_test = test_labels.size();
    Row pred_labels(n_test);
    for (size_t i = 0; i < n_test; ++i)
    {
        pred_labels[i] = (preds[i] >= 0.5) ? 1.0 : 0.0;
        if (pred_labels[i] == test_labels[i])
            correct_count++;
    }

    double accuracy = (double)correct_count / n_test * 100.0;
    std::cout << "Test Set Accuracy: " << accuracy << "%\n\n";

    exportResults(test.data, test.output, pred_labels, w);
}


void doMulticlassLogReg(t_csv& csv, t_csv& test)
{
    double mean = csv.means[csv.means.size()-1];
    double std = csv.stds[csv.stds.size()-1];
    Row& train_labels = denormalize(csv.output, mean, std);
    Row& test_labels = denormalize(test.output, mean, std);

    std::set<double> unique(train_labels.begin(), train_labels.end());
    size_t num_classes = unique.size();

    Matrix all_weights;
    std::vector<double> class_list(unique.begin(), unique.end());

    for (double target_class : class_list)
    {
        Row binary_labels(train_labels.size());
        for (size_t i = 0; i < train_labels.size(); i++)
            binary_labels[i] = (std::abs(train_labels[i] - target_class) < 1e-7) ? 1.0 : 0.0;
        all_weights.push_back(logisticRegression(csv.data, binary_labels));
    }

    size_t n_test = test_labels.size();
    int correct_count = 0;
    for (size_t i = 0; i < n_test; i++)
    {
        double max_prob = -1;
        double best_class = class_list[0];
        for (size_t k = 0; k < num_classes; k++) 
        {
            double z = 0.0;
            for (size_t j = 0; j < test.data[i].size(); j++)
                z += test.data[i][j] * all_weights[k][j];
            double prob = 1.0 / (1.0 + std::exp(-z));
            if (prob > max_prob) 
            {
                max_prob = prob;
                best_class = class_list[k];
            }
        }
        if (std::abs(best_class - test_labels[i]) < 1e-7)
            correct_count++;
    }

    double accuracy = (double) correct_count / n_test * 100.0;
    std::cout << "Correct Predictions: " << correct_count << "/" << n_test << "\n";
    std::cout << "Final Multiclass Accuracy: " << accuracy << "%\n";
}