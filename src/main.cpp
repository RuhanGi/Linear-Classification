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

int	main(int argc, char *argv[])
{
	if (argc != 2 || !argv)
		return (std::cerr << RED "Input One Argument!" RESET "\n", 1);

	try
	{
		t_csv csv = parseCSV(argv[1]);
		printStats(csv);
		doBasis(csv);
		Row w = doGradientDescent(csv, LAMBDA);
		Row preds = makePreds(csv.data, w);
		std::cout << "\n" << rSqr(csv.output, preds) << "\n";
		w = doGradientDescent(csv);
		preds = makePreds(csv.data, w);
		std::cout << "\n" << rSqr(csv.output, preds) << "\n";
	}
	catch (std::exception & e)
	{
		std::cerr << RED "ERROR: " << e.what() << RESET "\n";
		return (1);
	}

	return (0);
}
