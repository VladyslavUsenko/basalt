
#ifndef CONSOLE_BRIDGE_DLLAPI_H
#define CONSOLE_BRIDGE_DLLAPI_H

#ifdef CONSOLE_BRIDGE_STATIC_DEFINE
#  define CONSOLE_BRIDGE_DLLAPI
#  define CONSOLE_BRIDGE_NO_EXPORT
#else
#  ifndef CONSOLE_BRIDGE_DLLAPI
#    ifdef console_bridge_EXPORTS
        /* We are building this library */
#      define CONSOLE_BRIDGE_DLLAPI __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CONSOLE_BRIDGE_DLLAPI __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CONSOLE_BRIDGE_NO_EXPORT
#    define CONSOLE_BRIDGE_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CONSOLE_BRIDGE_DEPRECATED
#  define CONSOLE_BRIDGE_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef CONSOLE_BRIDGE_DEPRECATED_EXPORT
#  define CONSOLE_BRIDGE_DEPRECATED_EXPORT CONSOLE_BRIDGE_DLLAPI CONSOLE_BRIDGE_DEPRECATED
#endif

#ifndef CONSOLE_BRIDGE_DEPRECATED_NO_EXPORT
#  define CONSOLE_BRIDGE_DEPRECATED_NO_EXPORT CONSOLE_BRIDGE_NO_EXPORT CONSOLE_BRIDGE_DEPRECATED
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CONSOLE_BRIDGE_NO_DEPRECATED
#endif

#endif
