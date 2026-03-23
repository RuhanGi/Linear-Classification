/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/01/28 18:35:50 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/01/28 18:35:50 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Linear.hpp"

void filterBinaryClasses(t_csv& dataset)
{
    Matrix filtered_data;
    Row filtered_output;
    
    for (size_t i = 0; i < dataset.output.size(); ++i)
    {
        double label = dataset.output[i];
        if (std::abs(label - 0.0) < 1e-6 || std::abs(label - 1.0) < 1e-6)
        {
            filtered_data.push_back(dataset.data[i]);
            filtered_output.push_back(label);
        }
    }
    
    dataset.data = filtered_data;
    dataset.output = filtered_output;
    std::cout << "Filtered dataset to binary. New size: " << dataset.output.size() << " rows.\n";
}

int	main(int argc, char *argv[])
{
	if (argc != 2 || !argv)
		return (std::cerr << RED "Input One Argument!" RESET "\n", 1);

	try
	{
		t_csv csv = parseCSV(argv[1]);
		filterBinaryClasses(csv);
		printStats(csv);

		// t_csv test = doBasis(csv);
		// calcBayesian(csv, test);
		// doLogReg(csv, test);

		doGDA(csv);
	}
	catch (std::exception & e)
	{
		std::cerr << RED "ERROR: " << e.what() << RESET "\n";
		return (1);
	}

	return (0);
}
