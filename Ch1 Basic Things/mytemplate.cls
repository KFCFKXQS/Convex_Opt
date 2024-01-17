\NeedsTeXFormat{LaTeX2e}[1994/06/01]
\ProvidesClass{mytemplate}[2020/10/01 My custom class]
\LoadClass{article}

\RequirePackage[final]{neurips_2020}
\RequirePackage[utf8]{inputenc}
\RequirePackage{ctex}
\RequirePackage[T1]{fontenc}
\RequirePackage{hyperref}
\RequirePackage{url}
\RequirePackage{booktabs}
\RequirePackage{amsfonts}
\RequirePackage{amsmath}
\RequirePackage{nicefrac}
\RequirePackage{microtype}
\RequirePackage{indentfirst}
\RequirePackage{listings}
\RequirePackage{graphicx}
\RequirePackage{graphics}
\RequirePackage{float}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage{xcolor}
\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{amssymb}
\RequirePackage{mdframed}
\RequirePackage{tikz}
\lstset{
    language=Python, % 设置语言
    basicstyle=\ttfamily, % 设置字体族
    breaklines=true, % 自动换行
    keywordstyle=\bfseries\color{NavyBlue}, % 设置关键字为粗体，颜色为 NavyBlue
    morekeywords={}, % 设置更多的关键字，用逗号分隔
    emph={self}, % 指定强调词，如果有多个，用逗号隔开
    emphstyle=\bfseries\color{Rhodamine}, % 强调词样式设置
    commentstyle=\itshape\color{black!50!white}, % 设置注释样式，斜体，浅灰色
    stringstyle=\bfseries\color{PineGreen!90!black}, % 设置字符串样式
    columns=flexible,
    numbers=left, % 显示行号在左边
    numbersep=2em, % 设置行号的具体位置
    numberstyle=\footnotesize, % 缩小行号
    frame=single, % 边框
    framesep=1em, % 设置代码与边框的距离
    showstringspaces=false
}

% 定义边框样式
\mdfdefinestyle{theoremstyle}{%
    linecolor=gray,linewidth=1pt,%
    leftmargin=2em,rightmargin=2em,%
    backgroundcolor=gray!20,%
    innertopmargin=1em,innerbottommargin=1em,%
    splittopskip=\topskip,splitbottomskip=0.3\topskip,%
}

% 用mdframed包装定理环境
\surroundwithmdframed[style=theoremstyle]{theorem}
\surroundwithmdframed[style=theoremstyle]{proposition}
\surroundwithmdframed[style=theoremstyle]{definition}                   
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}

\theoremstyle{plain}
\newtheorem{proposition}{Proposition}

\theoremstyle{definition}
\newtheorem{definition}{Definition}
\setlength{\parindent}{2em}


\newcommand{\R}{\mathbb{R}}
