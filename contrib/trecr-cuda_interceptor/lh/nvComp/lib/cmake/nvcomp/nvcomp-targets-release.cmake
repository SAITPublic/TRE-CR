#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "nvcomp::nvcomp" for configuration "Release"
set_property(TARGET nvcomp::nvcomp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(nvcomp::nvcomp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libnvcomp.so"
  IMPORTED_SONAME_RELEASE "libnvcomp.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS nvcomp::nvcomp )
list(APPEND _IMPORT_CHECK_FILES_FOR_nvcomp::nvcomp "${_IMPORT_PREFIX}/lib/libnvcomp.so" )

# Import target "nvcomp::nvcomp_gdeflate" for configuration "Release"
set_property(TARGET nvcomp::nvcomp_gdeflate APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(nvcomp::nvcomp_gdeflate PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libnvcomp_gdeflate.so"
  IMPORTED_SONAME_RELEASE "libnvcomp_gdeflate.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS nvcomp::nvcomp_gdeflate )
list(APPEND _IMPORT_CHECK_FILES_FOR_nvcomp::nvcomp_gdeflate "${_IMPORT_PREFIX}/lib/libnvcomp_gdeflate.so" )

# Import target "nvcomp::nvcomp_bitcomp" for configuration "Release"
set_property(TARGET nvcomp::nvcomp_bitcomp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(nvcomp::nvcomp_bitcomp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libnvcomp_bitcomp.so"
  IMPORTED_SONAME_RELEASE "libnvcomp_bitcomp.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS nvcomp::nvcomp_bitcomp )
list(APPEND _IMPORT_CHECK_FILES_FOR_nvcomp::nvcomp_bitcomp "${_IMPORT_PREFIX}/lib/libnvcomp_bitcomp.so" )

# Import target "nvcomp::nvcomp_gdeflate_cpu" for configuration "Release"
set_property(TARGET nvcomp::nvcomp_gdeflate_cpu APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(nvcomp::nvcomp_gdeflate_cpu PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libnvcomp_gdeflate_cpu.so"
  IMPORTED_SONAME_RELEASE "libnvcomp_gdeflate_cpu.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS nvcomp::nvcomp_gdeflate_cpu )
list(APPEND _IMPORT_CHECK_FILES_FOR_nvcomp::nvcomp_gdeflate_cpu "${_IMPORT_PREFIX}/lib/libnvcomp_gdeflate_cpu.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
