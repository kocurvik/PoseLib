cmake -S . -B _build/ -DPYTHON_PACKAGE=ON -DCMAKE_INSTALL_PREFIX=_install
cmake --build _build/ --target install -j 8 --config=Release
cd _install/lib
copy PoseLib.lib libPoseLib.a
cd ../..
cmake --build _build/ --target pip-package
cmake --build _build/ --target install-pip-package