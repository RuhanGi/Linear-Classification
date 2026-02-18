/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   parse.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/01/30 17:04:14 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/01/30 17:04:14 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Linear.hpp"


double resolveToken(std::string token) {
    static const std::map<std::string, double> dict = {
        {"Iris-setosa", 0.0},
		{"Iris-versicolor", 1.0},
		{"Iris-virginica", 2.0},
		{"male", 0.0},
		{"female", 1.0},
		{"C", 0.0},
		{"Q", 1.0},
		{"S", 2.0},
    };

    auto it = dict.find(token);
    if (it != dict.end())
        return it->second;
    try {
        return std::stod(token);
    } catch (...) {}

	static std::map<std::string, double> dynamic_map;
    static double next_id = 1.0;

    if (dynamic_map.find(token) == dynamic_map.end()) {
        dynamic_map[token] = next_id++;
    }
    return dynamic_map[token];
}


std::vector<std::string> parseHeader(std::string line, char c) {
	std::vector<std::string> data;
	std::string str;
	std::istringstream lineStream(line);

	while (std::getline(lineStream, str, c))
		data.push_back(str);
	return data;
}


std::vector<std::string> splitLine(const std::string& line)
{
    std::vector<std::string> tokens;
    std::string token;
    bool in_quotes = false;

    for (char c : line)
    {
        if (c == '"')
            in_quotes = !in_quotes;
        else if (c == ',' && !in_quotes) {
            tokens.push_back(token);
            token.clear();
        } 
        else
            token += c;
    }
    tokens.push_back(token);
    return tokens;
}


Row processTokens(const std::vector<std::string>& raw_tokens, const std::vector<bool>& mask)
{
    Row row;
    size_t limit = std::min(raw_tokens.size(), mask.size());

    for (size_t i = 0; i < limit; i++)
    {
        if (mask[i])
		{
            if (!raw_tokens[i].empty())
                row.push_back(resolveToken(raw_tokens[i]));
            else
                row.push_back(0.0);
        }
    }
    return row;
}


void extractTarget(t_csv& csv, const std::string& target_name)
{
    if (csv.data.empty())
		return;

    int target_idx = -1;

    auto it = std::find(csv.headers.begin(), csv.headers.end(), target_name);
    if (it != csv.headers.end())
        target_idx = std::distance(csv.headers.begin(), it);
    else
        target_idx = csv.headers.size() - 1;

    if (target_idx < 0)
		return;

    csv.label = csv.headers[target_idx];
    csv.headers.erase(csv.headers.begin() + target_idx);
    csv.output.reserve(csv.data.size());
    for (Row& row : csv.data) {
        if (target_idx < (int)row.size()) {
            csv.output.push_back(row[target_idx]);
            row.erase(row.begin() + target_idx);
        } else
            csv.output.push_back(0.0);
    }
}


t_csv parseCSV(std::string fil)
{
	static std::vector<std::string> drop_cols = {"Id", "PassengerId", "Name", "Ticket", "Cabin"};
	std::ifstream file(fil.c_str());
	if (!file.is_open())
		throw std::runtime_error("Error: Could not open " + fil);

	t_csv csv;
	std::string line;
	std::vector<bool> col_mask;
	std::unordered_set<std::string> drop_set(drop_cols.begin(), drop_cols.end());

	if (std::getline(file, line))
	{
		std::vector<std::string> headers = parseHeader(line, ',');
        for (const auto& h : headers)
		{
            bool keep = (drop_set.find(h) == drop_set.end());
            col_mask.push_back(keep);
            if (keep)
				csv.headers.push_back(h);
        }
	}
	while (std::getline(file, line))
	{
		std::vector<std::string> raw_tokens = splitLine(line);
        Row row = processTokens(raw_tokens, col_mask);
        if (!row.empty())
            csv.data.push_back(std::move(row));
	}
	file.close();
	extractTarget(csv, "Survived");
	return csv;
}
