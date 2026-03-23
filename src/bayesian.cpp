/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   bayesian.cpp                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/03/12 17:17:45 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/03/12 17:17:45 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Linear.hpp"


void appendCSV(const std::string& filename, const std::string& new_header, Row& new_col)
{
    std::vector<std::string> lines;
    std::ifstream infile(filename);
    
    if (infile.is_open())
    {
        std::string line;
        while (std::getline(infile, line))
        {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            lines.push_back(line);
        }
        infile.close();
    }

    std::ofstream outfile(filename);
    if (!outfile.is_open())
        throw std::runtime_error("Could not open: " + filename);

    if (lines.empty()) {
        outfile << new_header << "\n";
        for (double val : new_col)
            outfile << val << "\n";
        outfile.close();
        return;
    }

    outfile << lines[0] << "," << new_header << "\n";

    size_t num_commas = std::count(lines[0].begin(), lines[0].end(), ',');
    size_t max_rows = std::max(lines.size() - 1, new_col.size());

    for (size_t i = 0; i < max_rows; ++i) {
        if (i + 1 < lines.size())
            outfile << lines[i + 1];
        else
            outfile << std::string(num_commas, ',');
        outfile << ",";
        if (i < new_col.size())
            outfile << new_col[i];
        outfile << "\n";
    }
    outfile.close();
}


Matrix calcSN(const Matrix& phi, double alpha, double beta)
{
    size_t M = phi[0].size();

    
    // this applies prior variance
    Matrix alph(M, Row(M, 0.0));
    for (size_t i = 0; i < M; i++)
        alph[i][i] = alpha;

    Matrix gram = transpose(phi) * phi;
    Matrix inv = alph + beta * gram;
    return invert(inv);
}


Row calcVars(double prec, Matrix phi, Matrix SN)
{
    size_t N = phi.size();
    Row vars(N);
    // for (size_t i = 0; i < N; i++)
    //     vars.push_back(prec + dot(phi[i] * SN, phi[i]));

    std::transform(std::execution::par, phi.begin(), phi.end(), vars.begin(),
        [prec, &SN](const Row& phi_i) {
            return prec + dot(phi_i * SN, phi_i);
        }
    );

    return vars;
}


void denormalize(Row& u, Row& vars, Row& truth, double mean, double std)
{
    for (size_t i = 0; i < u.size(); i++)
    {
        u[i] = u[i] * std + mean;
        vars[i] = std * std::sqrt(vars[i]); 
        truth[i] = truth[i] * std + mean;
    }
}


void exportPosterior(const Row& mN, const Matrix& SN, const std::string& filename)
{
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Could not open: " + filename);
    }

    outfile << "type,idx1,idx2,val\n";

    for (size_t i = 0; i < mN.size(); ++i)
        outfile << "mN," << i << ",0," << mN[i] << "\n";

    for (size_t i = 0; i < SN.size(); ++i)
        for (size_t j = 0; j < SN[i].size(); ++j)
            outfile << "SN," << i << "," << j << "," << SN[i][j] << "\n";
    
    outfile.close();
    std::cout << GREY << "Exported Posterior Weights to " << filename << RESET "\n"; 
}


void calcBayesian(t_csv& csv, t_csv& test, double alpha, double beta)
{
    Row preds;
    Matrix SN = calcSN(csv.data, alpha, beta);
    // note mN = SN (So^⁻¹m0 + B * phiT * t), but zero prior so:
    Row mN = beta * SN * (transpose(csv.data) * csv.output);
    Row u = test.data * mN;
    Row vars = calcVars(1.0 / beta, test.data, SN);
    denormalize(u, vars, test.output, test.means[test.data[0].size()], test.stds[test.data[0].size()]);

    Row train_u = csv.data * mN;
    Row train_vars(csv.data.size(), 0.0);
    Row train_true = csv.output;
    denormalize(train_u, train_vars, train_true, test.means[test.data[0].size()], test.stds[test.data[0].size()]);

    Row all_u = u;
    all_u.insert(all_u.end(), train_u.begin(), train_u.end());
    Row all_vars = vars;
    all_vars.insert(all_vars.end(), train_vars.begin(), train_vars.end());
    Row all_true = test.output;
    all_true.insert(all_true.end(), train_true.begin(), train_true.end());

    std::remove(OUTFILE);
    appendCSV(OUTFILE, "means", all_u);
    appendCSV(OUTFILE, "stds", all_vars);
    appendCSV(OUTFILE, "true", all_true);
    std::cout << GREY "File Creaated: " OUTFILE RESET "\n";

    exportPosterior(mN, SN, "Weights.csv");
}
