FILE(GLOB hdrs_base "*.h" )
FILE(GLOB srcs_base "*.c*")

FILE(GLOB hdrs ${hdrs_base}  )
FILE(GLOB srcs  ${srcs_base} )


ADD_LIBRARY(${PROJECT_NAME} ${srcs} ${hdrs})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR} ) 

SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES 				# create *nix style library versions + symbolic links
        DEFINE_SYMBOL DBOW_DSO_EXPORTS
	VERSION ${PROJECT_VERSION}
	SOVERSION ${PROJECT_SOVERSION}
	CLEAN_DIRECT_OUTPUT 1 					# allow creating static and shared libs without conflicts
	OUTPUT_NAME "${PROJECT_NAME}${PROJECT_DLLVERSION}" 	# avoid conflicts between library and binary target names
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${REQUIRED_LIBRARIES} ${OpenCV_LIBS})

