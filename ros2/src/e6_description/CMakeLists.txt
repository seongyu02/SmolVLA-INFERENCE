cmake_minimum_required(VERSION 3.8)
project(e6_description)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY urdf meshes launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  e6_description/joint_state_bridge_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
