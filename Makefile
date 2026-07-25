testfile= example_files/FASTIn_arf_coords.txt
all:
	python pyDatView.py $(testfile)

deb:
	python DEBUG.py

install:
	python setup.py install

dep:
	python -m pip install -e .

pull:
	git pull --recurse-submodules
update:pull


help:
	@echo "Available rules:"
	@echo "   all        run the standalone program"
	@echo "   install    install the python package in the system" 
	@echo "   dep        download the dependencies " 
	@echo "   pull       download the latest version " 
	@echo "   test       run the unit tests " 

test:
	python -m unittest discover -v tests
	python -m unittest discover -v pydatview/plugins/tests

clean:
	rm -rf __pycache__
	rm -rf *.egg-info
	rm -rf *.spec
	rm -rf build*
	rm -rf dist
	

pyexe:
	python -m PyInstaller --noconfirm --clean --windowed --onedir --name pyDatView --icon ressources/pyDatView.ico --add-data "ressources:ressources" pyDatView.py

version:
ifeq ($(OS),Windows_NT)
	@echo "Doing nothing"
else
	@sh _tools/setVersion.sh
endif

installer: pyexe

