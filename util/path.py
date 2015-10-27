from os import getcwd

def get_project_dir():
	'''
		Returns the project directory
	'''
	cwd = getcwd()
	return cwd[:cwd.rfind('src/')]
