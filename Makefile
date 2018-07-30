# Created from environment variables so modify per your system
# The output data storage directory
# OUTPUT_DIR=${WORK}/../tmp/data*

# Input file/list of files in Root
# ROOT_INPUT=${ROOT_SAMPLES}

clean:
	rm -rf DeepJetCore.egg-info/ build/ dist/ && make clean -C DeepJetCore/compiled

install:
	python setup.py build install

rebuild:
	make -C DeepJetCore/compiled

# these command are system-specific since they use environment variables
# convert:
#	convertFromRoot.py -i ${ROOT_SAMPLES} -o ${WORK}/../tmp/data-new -c TrainData_deepCSV
