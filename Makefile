clean:
	rm -rf DeepJetCore.egg-info/ build/ dist/ && make clean -C DeepJetCore/compiled

install:
	python setup.py build install

rebuild:
	make -C DeepJetCore/compiled

