# Karoo GP Pause Menu
# A text-based user interface for mid-run parameter configuration and population studies
# by Kai Staats, MSc; see LICENSE.md
# version 2.1.2

def pause(menu_dict):

	'''
	Pause the program execution and invok the user to make one or more valid options. 
		
	Called by: fx_karoo_gp
	
	Arguments required: menu_dict
	'''
	
	options = ['','?','help','i','m','g','s','db','ts','min','bal','l','pop','t','p','id','dir','load','w','cont','q']
	
	while True:
		try:
			menu = raw_input('\n\t\033[36m (pause) \033[0;0m')
			if menu not in options: raise ValueError()
			else: break
		except ValueError: print '\t\033[32m Select from the options given. Try again ...\033[0;0m'
		except KeyboardInterrupt: print '\n\t\033[32m Enter q to quit\033[0;0m'
		
	if menu == '': menu_dict['input_a'] = 'esc'; return menu_dict # bypass the following with ENTER
	
	elif menu == '?' or menu == 'help':
		print '\n\t\033[36mSelect one of the following options:\033[0;0m'
		print '\t\033[36m\033[1m i \t\033[0;0m engage Interactive display mode'
		print '\t\033[36m\033[1m m \t\033[0;0m engage Minimal display mode'
		print '\t\033[36m\033[1m g \t\033[0;0m engage Generation display mode'
		print '\t\033[36m\033[1m s \t\033[0;0m engage Silent display mode'
		print '\t\033[36m\033[1m db \t\033[0;0m engage De-Bug display mode'
		print ''
		print '\t\033[36m\033[1m ts \t\033[0;0m adjust tournament size'
		print '\t\033[36m\033[1m min \t\033[0;0m adjust minimum number of nodes'
		# print '\t\033[36m\033[1m max \t\033[0;0m adjust maximum Tree depth' # future option
		print '\t\033[36m\033[1m bal \t\033[0;0m adjust balance of genetic operators'
		print ''
		print '\t\033[36m\033[1m l \t\033[0;0m list Trees with leading fitness scores'
		print '\t\033[36m\033[1m pop \t\033[0;0m list Trees in current population'
		print '\t\033[36m\033[1m t \t\033[0;0m evaluate a single Tree against the test data'
		print '\t\033[36m\033[1m p \t\033[0;0m print a single Tree to screen'
		print ''
		print '\t\033[36m\033[1m id \t\033[0;0m display current generation ID'
		print '\t\033[36m\033[1m dir \t\033[0;0m display current working directory'
		# print '\t\033[36m\033[1m load \t\033[0;0m load population_s (seed) to replace population_a (current)'
		print '\t\033[36m\033[1m w \t\033[0;0m write the evolving population_b to disk'
		print ''
		print '\t\033[36m\033[1m cont \t\033[0;0m add generations and continue your run'
		print '\t\033[36m\033[1m q \t\033[0;0m quit Karoo GP'
		
		if menu_dict['gen_id'] == menu_dict['gen_max']: print '\n\t Your GP run is complete. You may \033[36m\033[1madd\033[0;0m generations and \033[36m\033[1mcont\033[0;0minue, or \033[36m\033[1mq\033[0;0muit.'
		else: print '\n\t You are mid-run. You may \033[36m\033[1m cont\033[0;0minue or \033[36m\033[1mq\033[0;0muit without saving population_b.'
		
	elif menu == 'i': menu_dict['display'] = 'i'; print '\n\t Interactive display mode engaged (for control freaks)'
	elif menu == 'm': menu_dict['display'] = 'm'; print '\n\t Minimal display mode engaged (for recovering control freaks)'
	elif menu == 'g': menu_dict['display'] = 'g'; print '\n\t Generation display mode engaged (for GP gurus)'
	elif menu == 's': menu_dict['display'] = 's'; print '\n\t Silent display mode engaged (for zen masters)'
	elif menu == 'db': menu_dict['display'] = 'db'; print '\n\t De-Bug display mode engaged (for evolutionary biologists)'
	
	elif menu == 'ts': # adjust the tournament size
		while True:
			try:
				print '\n\t The current tournament size is:', menu_dict['tourn_size']
				query = raw_input('\t Adjust the tournament size (suggest 7 for each 100): ')
				if query not in str(range(2,menu_dict['tree_pop_max'] + 1)) or query == '0' or query == '1': raise ValueError() # not ideal 20170918
				elif query == '': break
				else: menu_dict['tourn_size'] = int(query); break
			except ValueError: print '\n\t\033[32m Enter a number from 2 including %s. Try again ...\033[0;0m' %str(menu_dict['tree_pop_max'])
			except KeyboardInterrupt: print '\n\t\033[32m Enter q to quit\033[0;0m'
	
	elif menu == 'min': # adjust the minimum number of nodes per Tree
		# max_nodes = 2**(tree_depth_base +1) - 1 # NEED TO calc to replace upper limit in range but tree_depth_base is not global - 2018 04/22
		while True:
			try:
				print '\n\t The current minimum number of nodes is:', menu_dict['tree_depth_min']
				query = raw_input('\t Adjust the minimum number of nodes for all Trees (min 3): ')
				if query not in str(range(3,1000)) or query == '0' or query == '1' or query == '2': raise ValueError() # not ideal 20170918
				elif query == '': break
				else: menu_dict['tree_depth_min'] = int(query); break
			except ValueError: print '\n\t\033[32m Enter a number from 3 including 1000. Try again ...\033[0;0m'
			except KeyboardInterrupt: print '\n\t\033[32m Enter q to quit\033[0;0m'

	# NEED TO ADD: adjustable tree_depth_max
	#elif menu == 'max': # adjust the maximum Tree depth
	#
	#	while True:
	#		try:
	#			print '\n\t The current \033[3madjusted\033[0;0m maximum Tree depth is:', gp.tree_depth_max
	#			query = raw_input('\n\t Adjust the global maximum Tree depth to (1 ... 10): ')
	#			if query not in str(range(1,11)): raise ValueError()
	#			if query < gp.tree_depth_max:
	#				print '\n\t\033[32m This value is less than the current value.\033[0;0m'
	#				conf = raw_input('\n\t Are you ok with this? (y/n) ')
	#				if conf == 'n': break
	#		except ValueError: print '\n\t\033[32m Enter a number from 1 including 10. Try again ...\033[0;0m'
	#		except KeyboardInterrupt: print '\n\t\033[32m Enter q to quit\033[0;0m'
	
	elif menu == 'bal': # adjust the balance of genetic operators'
		print '\n\t The current balance of genetic operators is:'
		print '\t\t Reproduction:', menu_dict['evolve_repro']; tmp_repro = menu_dict['evolve_repro']
		print '\t\t Point Mutation:', menu_dict['evolve_point']; tmp_point = menu_dict['evolve_point']
		print '\t\t Branch Mutation:', menu_dict['evolve_branch']; tmp_branch = menu_dict['evolve_branch']
		print '\t\t Crossover:', menu_dict['evolve_cross'], '\n'; tmp_cross = menu_dict['evolve_cross']
				
		while True:
			try:
				query = raw_input('\t Enter quantity of Trees to be generated by Reproduction: ')
				if query not in str(range(0,1000)): raise ValueError()
				elif query == '': break
				else: tmp_repro = int(query); break
			except ValueError: print '\n\t\033[32m Enter a number from 0 including %s. Try again ...\033[0;0m' %str(menu_dict['tree_pop_max'])
			except KeyboardInterrupt: print '\n\t\033[32m Enter q to quit\033[0;0m'
			
		while True:
			try:
				query = raw_input('\t Enter quantity of Trees to be generated by Point Mutation: ')
				if query not in str(range(0,1000)): raise ValueError()
				elif query == '': break
				else: tmp_point = int(query); break
			except ValueError: print '\n\t\033[32m Enter a number from 0 including %s. Try again ...\033[0;0m' %str(menu_dict['tree_pop_max'])
			except KeyboardInterrupt: print '\n\t\033[32m Enter q to quit\033[0;0m'
			
		while True:
			try:
				query = raw_input('\t Enter quantity of Trees to be generated by Branch Mutation: ')
				if query not in str(range(0,1000)): raise ValueError()
				elif query == '': break
				else: tmp_branch = int(query); break
			except ValueError: print '\n\t\033[32m Enter a number from 0 including %s. Try again ...\033[0;0m' %str(menu_dict['tree_pop_max'])
			except KeyboardInterrupt: print '\n\t\033[32m Enter q to quit\033[0;0m'
			
		while True:
			try:
				query = raw_input('\t Enter quantity of Trees to be generated by Crossover: ')
				if query not in str(range(0,1000)): raise ValueError()
				elif query == '': break
				else: tmp_cross = int(query); break
			except ValueError: print '\n\t\033[32m Enter a number from 0 including %s. Try again ...\033[0;0m' %str(menu_dict['tree_pop_max'])
			except KeyboardInterrupt: print '\n\t\033[32m Enter q to quit\033[0;0m'
			
		if tmp_repro + tmp_point + tmp_branch + tmp_cross != menu_dict['tree_pop_max']:
			print '\n\t The sum of the above does not equal %s. Try again ...' %str(menu_dict['tree_pop_max'])
			
		else:
			print '\n\t The revised balance of genetic operators is:'
			print '\t\t Reproduction:', tmp_repro; menu_dict['evolve_repro'] = tmp_repro
			print '\t\t Point Mutation:', tmp_point; menu_dict['evolve_point'] = tmp_point
			print '\t\t Branch Mutation:', tmp_branch; menu_dict['evolve_branch'] = tmp_branch
			print '\t\t Crossover:', tmp_cross; menu_dict['evolve_cross'] = tmp_cross
			
	elif menu == 'l': # display dictionary of Trees with the best fitness score
		print '\n\t The leading Trees and their associated expressions are:'
		for n in sorted(menu_dict['fittest_dict']): print '\t ', n, ':', menu_dict['fittest_dict'][n]
		
	elif menu == 'pop': # list Trees in the current population
		if menu_dict['gen_id'] == 1: menu_dict['input_a'] = 'pop_a'
		else: menu_dict['input_a'] = 'pop_b'
		
	elif menu == 't': # evaluate a Tree against the TEST data
		if menu_dict['gen_id'] == 1: print '\n\t\033[32m You cannot evaluate the foundation population. Be patient ...\033[0;0m'
		
		else: # gen_id > 1
			while True:
				try:
					query = raw_input('\n\t Select a Tree to test: ')
					if query not in str(range(1, menu_dict['pop_b_len'])) or query == '0': raise ValueError()
					elif query == '': break
					else: menu_dict['input_a'] = 'test'; menu_dict['input_b'] = int(query); break
				except ValueError: print '\n\t\033[32m Enter a number from 1 including %s. Try again ...\033[0;0m' %str(menu_dict['pop_b_len'] - 1)
				except KeyboardInterrupt: print '\n\t\033[32m Enter q to quit\033[0;0m'
				
	elif menu == 'p': # print a Tree to screen -- NEED TO ADD: SymPy graphical print option
	
		if menu_dict['gen_id'] == 1:
			while True:
				try:
					query = raw_input('\n\t Select a Tree to print: ')
					if query not in str(range(1, menu_dict['pop_a_len'])) or query == '0': raise ValueError()
					elif query == '': break
					else: menu_dict['input_a'] = 'print_a'; menu_dict['input_b'] = int(query); break
				except ValueError: print '\n\t\033[32m Enter a number from 1 including %s. Try again ...\033[0;0m' %str(menu_dict['pop_a_len'] - 1)
				except KeyboardInterrupt: print '\n\t\033[32m Enter q to quit\033[0;0m'
				
		else: # gen_id > 1
			while True:
				try:
					query = raw_input('\n\t Select a Tree to print: ')
					if query not in str(range(1, menu_dict['pop_b_len'])) or query == '0': raise ValueError()
					elif query == '': break
					else: menu_dict['input_a'] = 'print_b'; menu_dict['input_b'] = int(query); break
				except ValueError: print '\n\t\033[32m Enter a number from 1 including %s. Try again ...\033[0;0m' %str(menu_dict['pop_b_len'] - 1)
				except KeyboardInterrupt: print '\n\t\033[32m Enter q to quit\033[0;0m'
				
	elif menu == 'id': print '\n\t The current generation is:', menu_dict['gen_id']
	
	elif menu == 'dir': print '\n\t The current working directory is:', menu_dict['path']
	
	# NEED TO review and fix
	elif menu == 'load': # load population_s to replace population_a
		while True:
			try:
				query = raw_input('\n\t Overwrite the current population with population_s? (y/n) ')
				if query not in ['y','n']: raise ValueError()
				if query == 'y': menu_dict['input_a'] = 'load'; break
				elif query == 'n': break
			except ValueError: print '\n\t\033[32m Enter (y)es or (n)o. Try again ...\033[0;0m'
			except KeyboardInterrupt: print '\n\t\033[32m Enter q to quit\033[0;0m'
			
	elif menu == 'w': # write the evolving population_b to disk
		if menu_dict['gen_id'] > 1: menu_dict['input_a'] = 'write'
		else: print '\n\t\033[36m The evolving population_b does not yet exist\033[0;0m'
		
	elif menu == 'cont': # continue a GP run
		if menu_dict['gen_id'] == menu_dict['gen_max']:
			while True:
				try:
					query = raw_input('\n\t\033[3m You are at the end of your run.\033[0;0m\n\t Add more generations to continue (1-100 or ENTER to escape): ')
					if query not in str(range(1,101)) or query == '0': raise ValueError()
					elif query == '': break
					else: menu_dict['input_a'] = 'cont'; menu_dict['input_b'] = int(query); break
				except ValueError: print '\n\t\033[32m Enter a number from 1 including 100. Try again ...\033[0;0m'
				except KeyboardInterrupt: print '\n\t\033[32m Enter q to quit\033[0;0m'
				
		else: menu_dict['input_a'] = 'cont'
		
	elif menu == 'q': # quit
		while True:
			try:
				query = raw_input('\n\t Quit Karoo GP? (y/n) ')
				if query not in ['y','n','']: raise ValueError()
				if query == 'y': menu_dict['input_a'] = 'quit'; break
				else: break
			except ValueError: print '\n\t\033[32m Enter (y)es or (n)o. Try again ...\033[0;0m'
			except KeyboardInterrupt: print '\n\t\033[32m Enter (y)es or (n)o. Try again ...\033[0;0m'
			
	return menu_dict
	

