#
# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://gitlab.com/VladyslavUsenko/basalt.git
#
# Copyright (c) 2019-2021, Vladyslav Usenko and Nikolaus Demmel.
# All rights reserved.
#
screenread_sty = r"""
\ProvidesPackage{screenread}
% Copyright (C) 2012 John Collins, collins@phys.psu.edu
% License: LPPL 1.2

% Note:  To avoid compatibility issues between geometry and at least one
% class file, it may be better to set all the dimensions by hand.

%  20 Nov 2014  - use `pageBreakSection` instead of clobbering `section`
%               - increase longest page size to 575cm
%               - make top, right, and left margins something sensible and
%                 a bit more aesthetically pleasing
%  24 Jan 2012  Argument to \SetScreen is screen width
%  23 Jan 2012  Remove package showlayout
%  22 Jan 2012  Initial version, based on ideas in
%               B. Veytsman amd M. Ware, Tugboat 32 (2011) 261.

\RequirePackage{everyshi}
\RequirePackage{geometry}

%=======================

\pagestyle{empty}

\EveryShipout{%
    \pdfpageheight=\pagetotal
    \advance\pdfpageheight by 2in
    \advance\pdfpageheight by \topmargin
    \advance\pdfpageheight by \textheight % This and next allow for footnotes
    \advance\pdfpageheight by -\pagegoal
}

\AtEndDocument{\pagebreak}

\def\pageBreakSection{\pagebreak\section}

\newlength\screenwidth
\newlength{\savedscreenwidth}

\newcommand\SetScreen[1]{%
  % Argument #1 is the screen width.
  % Set appropriate layout parameters, with only a little white space
  %   around the text.
  \setlength\screenwidth{#1}%
  \setlength\savedscreenwidth{#1}%
  \setlength\textwidth{#1}%
  \addtolength\textwidth{-2cm}%
  \geometry{layoutwidth=\screenwidth,
            paperwidth=\screenwidth,
            textwidth=\textwidth,
            layoutheight=575cm,
            paperheight=575cm,
            textheight=575cm,
            top=1cm,
            left=1cm,
            right=1cm,
            hcentering=true
  }%
}

\newcommand\SetPageScreenWidth[1]{%
  \setlength\savedscreenwidth{\screenwidth}%
  \setlength\screenwidth{#1}%
  \pdfpagewidth\screenwidth%
  \setlength\textwidth{\screenwidth}%
  \addtolength\textwidth{-2cm}%
}

\newcommand\RestorePageScreenWidth{%
  \setlength\screenwidth{\savedscreenwidth}%
  \pdfpagewidth\screenwidth%
  \setlength\textwidth{\screenwidth}%
  \addtolength\textwidth{-2cm}%
}


% Compute a reasonable default screen width, and set it
\setlength\screenwidth{\textwidth}
\addtolength\screenwidth{1cm}
\SetScreen{\screenwidth}

\endinput

"""
