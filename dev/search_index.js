var documenterSearchIndex = {"docs":
[{"location":"api/#Available-functions","page":"API","title":"Available functions","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"","category":"page"},{"location":"api/","page":"API","title":"API","text":"CurrentModule = CYAxiverse","category":"page"},{"location":"api/#CYAxiverse.filestructure","page":"API","title":"CYAxiverse.filestructure","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"CYAxiverse.filestructure","category":"page"},{"location":"api/#CYAxiverse.filestructure","page":"API","title":"CYAxiverse.filestructure","text":"CYAxiverse.filestructure\n\nThis module sets up the structure of the database, identifying where to locate data / plot files etc\n\n\n\n\n\n","category":"module"},{"location":"api/","page":"API","title":"API","text":"Modules = [CYAxiverse.filestructure]\nPages = [\"filestructure.jl\"]","category":"page"},{"location":"api/#CYAxiverse.filestructure.Kfile","page":"API","title":"CYAxiverse.filestructure.Kfile","text":"Kfile(h11,tri,cy)\n\nLoads Kähler metric specified by geometry index.\n\nwarning: Warning\nDeprecated\n\n\n\n\n\n","category":"function"},{"location":"api/#CYAxiverse.filestructure.Lfile","page":"API","title":"CYAxiverse.filestructure.Lfile","text":"Lfile(h11,tri,cy)\n\nLoads instanton energy scales specified by geometry index.\n\nwarning: Warning\nDeprecated\n\n\n\n\n\n","category":"function"},{"location":"api/#CYAxiverse.filestructure.Qfile","page":"API","title":"CYAxiverse.filestructure.Qfile","text":"Qfile(h11,tri,cy)\n\nLoads instanton charge matrix specified by geometry index.\n\nwarning: Warning\nDeprecated\n\n\n\n\n\n","category":"function"},{"location":"api/#CYAxiverse.filestructure.cyax_file","page":"API","title":"CYAxiverse.filestructure.cyax_file","text":"cyax_file(h11,tri,cy)\n\nPath to data file – will contain all data that relates to geometry index.\n\n\n\n\n\n","category":"function"},{"location":"api/#CYAxiverse.filestructure.data_dir-Tuple{}","page":"API","title":"CYAxiverse.filestructure.data_dir","text":"data_dir()\n\nCreates/reads data directory\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.filestructure.geom_dir","page":"API","title":"CYAxiverse.filestructure.geom_dir","text":"geom_dir(h11,tri,cy)\n\nDefines file directories for data specified by geometry index.\n\n\n\n\n\n","category":"function"},{"location":"api/#CYAxiverse.filestructure.h11lst","page":"API","title":"CYAxiverse.filestructure.h11lst","text":"h11lst(min,max)\n\nLoads geometry indices between h^11 in (mathrmminmathrmmax\n\n\n\n\n\n","category":"function"},{"location":"api/#CYAxiverse.filestructure.localARGS-Tuple{}","page":"API","title":"CYAxiverse.filestructure.localARGS","text":"localARGS()\n\nLoad key for data dir – key should be in ol_DB\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.filestructure.log_dir-Tuple{}","page":"API","title":"CYAxiverse.filestructure.log_dir","text":"log_dir()\n\nCreates/reads log directory\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.filestructure.logcreate-Tuple{String}","page":"API","title":"CYAxiverse.filestructure.logcreate","text":"logcreate(l)\n\nCreates logfile\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.filestructure.logfile-Tuple{}","page":"API","title":"CYAxiverse.filestructure.logfile","text":"logfile()\n\nReturns path of logfile in format data_dir()/logs/YYYY:MM:DD:T00:00:00.000log.out\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.filestructure.minfile","page":"API","title":"CYAxiverse.filestructure.minfile","text":"minfile(h11,tri,cy)\n\nPath to file containing minimization data.\n\n\n\n\n\n","category":"function"},{"location":"api/#CYAxiverse.filestructure.np_path-Tuple{}","page":"API","title":"CYAxiverse.filestructure.np_path","text":"np_path()\n\nWalks through data_dir() and returns list of data paths and matrix of [h11; tri; cy]. Saves in h5 file paths_cy.h5\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.filestructure.ol_DB-Tuple{Any}","page":"API","title":"CYAxiverse.filestructure.ol_DB","text":"ol_DB(args)\n\nDefine dict of directories for data read/write\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.filestructure.paths_cy-Tuple{}","page":"API","title":"CYAxiverse.filestructure.paths_cy","text":"paths_cy()\n\nLoads / generates paths_cy.h5 which contains the explicit locations and also [h11; tri; cy] indices of the geometries already saved.\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.filestructure.plots_dir-Tuple{}","page":"API","title":"CYAxiverse.filestructure.plots_dir","text":"plots_dir()\n\nCreates/reads a directory for plots\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.filestructure.present_dir-Tuple{}","page":"API","title":"CYAxiverse.filestructure.present_dir","text":"present_dir()\n\nReturns the present data directory using localARGS\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.generate","page":"API","title":"CYAxiverse.generate","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"CYAxiverse.generate","category":"page"},{"location":"api/#CYAxiverse.generate","page":"API","title":"CYAxiverse.generate","text":"CYAxiverse.generate\n\nThis is where most of the functions are defined.\n\n\n\n\n\n","category":"module"},{"location":"api/","page":"API","title":"API","text":"Modules = [CYAxiverse.generate]\nPages = [\"generate.jl\"]","category":"page"},{"location":"api/#CYAxiverse.generate.LQtildebar-Tuple{Matrix{Float64}, Matrix{Int64}}","page":"API","title":"CYAxiverse.generate.LQtildebar","text":"LQtildebar(L,Q; threshold)\n\nCompute the linearly independent leading instantons that generate the axion potential, including any subleading instantons that are within threshold of their basis instanton.\n\n#Examples\n\njulia> h11,tri,cy = 8, 108, 1;\njulia> pot_data = CYAxiverse.read.potential(h11,tri,cy);\njulia> vacua_data = CYAxiverse.generate.LQtildebar(pot_data[\"L\"],pot_data[\"Q\"]; threshold=0.5)\nDict{String, Matrix{Float64}}(\n\"Ltilde\" => 2×9 Matrix{Float64}:\n   1.0       1.0       1.0       1.0     …     1.0       1.0       1.0       1.0\n -46.5338  -49.2362  -57.3519  -92.6082     -136.097  -163.303  -179.634  -163.303\n\n\"Lbar\" => 2×12 Matrix{Float64}:\n   1.0      1.0        1.0       1.0    …    -1.0      -1.0       1.0       1.0\n -90.414  -98.5299  -101.232  -133.787     -177.681  -177.681  -179.978  -179.978\n\n\"Qbar\" => 8×12 Matrix{Float64}:\n  0.0   0.0   0.0   2.0   2.0   2.0  1.0   0.0   0.0   0.0   0.0   0.0\n  0.0   0.0   0.0   0.0   0.0   0.0  0.0   0.0   0.0   0.0   0.0   0.0\n  0.0   0.0   0.0  -1.0  -1.0  -1.0  0.0  -1.0   1.0   0.0   1.0   0.0\n  0.0   0.0   0.0  -1.0  -1.0  -1.0  0.0   1.0   0.0   1.0   0.0   1.0\n  0.0   0.0   0.0   0.0   0.0   0.0  0.0   1.0   0.0   0.0   0.0   0.0\n  1.0   1.0   0.0   0.0   1.0   1.0  0.0   0.0  -1.0  -1.0   0.0   0.0\n -1.0   0.0   1.0   1.0   0.0   1.0  0.0   0.0   0.0   0.0  -1.0  -1.0\n  0.0  -1.0  -1.0  -1.0  -1.0  -2.0  0.0   0.0   0.0   0.0   0.0   0.0\n\n\"α\" => 12×8 Matrix{Float64}:\n  1.0  -1.0   0.0  0.0   0.0  0.0  0.0  0.0\n  1.0   0.0  -1.0  0.0   0.0  0.0  0.0  0.0\n  0.0   1.0  -1.0  0.0   0.0  0.0  0.0  0.0\n -1.0   0.0   0.0  1.0   0.0  0.0  0.0  0.0\n  0.0  -1.0   0.0  1.0   0.0  0.0  0.0  0.0\n  0.0   0.0  -1.0  1.0   0.0  0.0  0.0  0.0\n  0.0   0.0   0.0  0.0   0.0  0.0  0.0  0.0\n  0.0   0.0   0.0  0.0  -1.0  1.0  1.0  0.0\n -1.0   0.0   0.0  0.0   1.0  0.0  0.0  0.0\n -1.0   0.0   0.0  0.0   0.0  1.0  0.0  0.0\n  0.0  -1.0   0.0  0.0   1.0  0.0  0.0  0.0\n  0.0  -1.0   0.0  0.0   0.0  1.0  0.0  0.0\n\n\"Qtilde\" => 8×9 Matrix{Float64}:\n 0.0  0.0  0.0   2.0  0.0  0.0  0.0  0.0   0.0\n 0.0  0.0  0.0   0.0  0.0  0.0  0.0  1.0   0.0\n 0.0  0.0  0.0  -1.0  1.0  0.0  0.0  0.0  -1.0\n 0.0  0.0  0.0  -1.0  0.0  1.0  0.0  0.0   1.0\n 0.0  0.0  0.0   0.0  0.0  0.0  1.0  0.0   1.0\n 1.0  0.0  0.0   1.0  0.0  0.0  0.0  0.0   0.0\n 0.0  1.0  0.0   1.0  0.0  0.0  0.0  0.0   0.0\n 0.0  0.0  1.0  -1.0  0.0  0.0  0.0  0.0   0.0\n )\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.generate.constants-Tuple{}","page":"API","title":"CYAxiverse.generate.constants","text":"constants()\n\nLoads constants:\n\nReduced Planck Mass = 2.435 × 10^18\nHubble = 2.13 × 0.7 × 10^-33\nlog2pi = log10(2π)\n\nas Dict{String,ArbFloat}\n\n#Examples\n\njulia> const_data = CYAxiverse.generate.constants()\nDict{String, ArbNumerics.ArbFloat{128}} with 3 entries:\n  \"MPlanck\" => 2435000000000000000.0\n  \"log2π\"   => 0.7981798683581150521959557408991\n  \"Hubble\"  => 1.490999999999999999287243983194e-33\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.generate.gauss_diff-Tuple{Float64}","page":"API","title":"CYAxiverse.generate.gauss_diff","text":"gauss_diff(z)\n\nComputes the difference of 2 numbers in (natural) log-space using the definition here.\n\n#Examples\n\njulia> CYAxiverse.generate.gauss_diff(10.)\n9.99995459903963\n\njulia> CYAxiverse.generate.gauss_diff(1000.)\n1000.0\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.generate.gauss_log_split-Tuple{Vector{Int64}, Vector{Float64}}","page":"API","title":"CYAxiverse.generate.gauss_log_split","text":"gauss_log_split(sign, log)\n\nAlgorithm to compute Gaussian logarithms, as detailed here.\n\n#Examples\n\njulia> CYAxiverse.generate.gauss_diff(10.)\n9.99995459903963\n\njulia> CYAxiverse.generate.gauss_diff(1000.)\n1000.0\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.generate.gauss_sum-Tuple{Float64}","page":"API","title":"CYAxiverse.generate.gauss_sum","text":"gauss_sum(z)\n\nComputes the addition of 2 numbers in (natural) log-space using the definition here.\n\n#Examples\n\njulia> CYAxiverse.generate.gauss_sum(10.)\n10.000045398899218\n\njulia> CYAxiverse.generate.gauss_sum(1000.)\n1000.0\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.generate.hp_spectrum-Tuple{LinearAlgebra.Hermitian{Float64, Matrix{Float64}}, Matrix{Float64}, Matrix{Int64}}","page":"API","title":"CYAxiverse.generate.hp_spectrum","text":"hp_spectrum(K,L,Q; prec=5_000)\n\nUses potential data generated by CYTools (or randomly generated) to compute axion spectra – masses, quartic couplings and decay constants – to high precision.\n\n#Examples\n\njulia> pot_data = CYAxiverse.read.potential(4,10,1)\njulia> hp_spec_data = CYAxiverse.generate.hp_spectrum(pot_data[\"K\"], pot_data[\"L\"], pot_data[\"Q\"])\nDict{String, Any} with 12 entries:\n    \"msign\" => []\n    \"m\" => []\n    \"fK\" => []\n    \"fpert\" => []\n    \"λselfsign\" => []\n    \"λself\" => []\n    \"λ31_i\" => []\n    \"λ31sign\" => []\n    \"λ31\" => []\n    \"λ22_i\" => []\n    \"λ22sign\" => []\n    \"λ22\" => []\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.generate.hp_spectrum_save","page":"API","title":"CYAxiverse.generate.hp_spectrum_save","text":"hp_spectrum_save(h11,tri,cy)\n\n\n\n\n\n","category":"function"},{"location":"api/#CYAxiverse.generate.orth_basis-Tuple{Matrix}","page":"API","title":"CYAxiverse.generate.orth_basis","text":"orth_basis(Q)\n\nTakes a set of vectors (columns of Q) and constructs an orthonormal basis\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.generate.pq_spectrum-Tuple{LinearAlgebra.Hermitian{Float64, Matrix{Float64}}, Matrix{Float64}, Matrix{Int64}}","page":"API","title":"CYAxiverse.generate.pq_spectrum","text":"pq_spectrum(K,L,Q)\n\nUses a modified version of the algorithm outlined in the PQ Axiverse paper (Appendix A) to compute the masses and decay constants.\n\nnote: Note\nThe off-diagonal elements of the quartic self-interaction tensor are not yet included in this computation\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.generate.pseudo_K","page":"API","title":"CYAxiverse.generate.pseudo_K","text":"pseudo_K(h11,tri,cy=1)\n\nRandomly generates an h11 × h11 Hermitian matrix with positive definite eigenvalues. \n\n#Examples\n\njulia> K = CYAxiverse.generate.pseudo_K(4,10,1)\n4×4 Hermitian{Float64, Matrix{Float64}}:\n 2.64578  2.61012  0.91203  2.27339\n 2.61012  3.89684  2.22451  1.93356\n 0.91203  2.22451  2.94717  1.58126\n 2.27339  1.93356  1.58126  4.85208\n\njulia> eigen(K).values\n4-element Vector{Float64}:\n 0.17629073145135896\n 1.8632009739875723\n 2.7425362219513487\n 9.559840749713599\n\n\n\n\n\n","category":"function"},{"location":"api/#CYAxiverse.generate.pseudo_L","page":"API","title":"CYAxiverse.generate.pseudo_L","text":"pseudo_L(h11,tri,cy=1;log=true)\n\nRandomly generates a h11+4+C(h11+4,2)-length hierarchical list of instanton scales, similar to those found in the KS Axiverse.  Option for (sign,log10) or full precision.\n\n#Examples\n\njulia> CYAxiverse.generate.pseudo_L(4,10)\n36×2 Matrix{Float64}:\n  1.0     0.0\n  1.0    -4.0\n  1.0    -8.0\n  1.0   -12.0\n  1.0   -16.0\n  1.0   -20.0\n  1.0   -24.0\n  1.0   -28.0\n -1.0   -29.4916\n  1.0   -33.8515\n  ⋮\n  1.0  -133.665\n -1.0  -138.951\n\njulia> CYAxiverse.generate.pseudo_L(4,10,log=false)\n36-element Vector{ArbNumerics.ArbFloat}:\n  1.0\n  0.0001\n  1.0e-8\n  1.0e-12\n  1.0e-16\n  1.0e-20\n  1.0e-24\n  1.0e-28\n -1.462574279422558833057690597964e-31\n -2.381595397961591074099629406235e-34\n  ⋮\n  3.796809523142314798130344022481e-134\n -3.173000613781491329619833894919e-138\n\n\n\n\n\n","category":"function"},{"location":"api/#CYAxiverse.generate.pseudo_Q","page":"API","title":"CYAxiverse.generate.pseudo_Q","text":"pseudo_Q(h11,tri,cy=1)\n\nRandomly generates an instanton charge matrix that takes the same form as those found in the KS Axiverse, namely I(h11) with 4 randomly filled rows and the cross-terms, i.e. an h11+4+C(h11+4,2) × h11 integer matrix.\n\n#Examples\n\njulia> CYAxiverse.generate.pseudo_Q(4,10,1)\n36×4 Matrix{Int64}:\n  1   0   0   0\n  0   1   0   0\n  0   0   1   0\n  0   0   0   1\n  1   4  -3   5\n -5  -4  -2   4\n  4   5   3  -2\n -5   2  -3  -3\n  ⋮\n -9  -9  -5   6\n  0  -6   1   7\n  9   3   6   1\n\n\n\n\n\n","category":"function"},{"location":"api/#CYAxiverse.generate.vacua-Tuple{Matrix{Float64}, Matrix{Int64}}","page":"API","title":"CYAxiverse.generate.vacua","text":"vacua(L,Q; threshold)\n\nCompute the number of vacua given an instanton charge matrix Q and 2-column matrix of instanton scales L (in the form [sign; exponent]) and a threshold for:\n\nfracLambda_aLambda_j\n\ni.e. is the instanton contribution large enough to affect the minima.\n\nFor small systems (Nax<=50) the algorithm computes the ratio of volumes of the fundamental domain of the leading potential and the full potential.\n\nFor larger systems, the algorithm only computes the volume of the fundamental domain of the leading potential.\n\n#Examples\n\njulia> using CYAxiverse\njulia> h11,tri,cy = 10,20,1;\njulia> pot_data = CYAxiverse.read.potential(h11,tri,cy);\njulia> vacua_data = CYAxiverse.generate.vacua(pot_data[\"L\"],pot_data[\"Q\"])\nDict{String, Any} with 3 entries:\n  \"θ∥\"     => Rational[1//1 0//1 … 0//1 0//1; 0//1 1//1 … 0//1 0//1; … ; 0//1 0//1 … 1//1 0//1; 0//1 0//1 … 0//1 1//1]\n  \"vacua\"  => 3\n  \"Qtilde\" => [0 0 … 1 0; 0 0 … 0 0; … ; 1 1 … 0 0; 0 0 … 0 0]\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.generate.vacua_MK-Tuple{Int64, Int64, Int64}","page":"API","title":"CYAxiverse.generate.vacua_MK","text":"vacua_MK(L,Q; threshold=1e-2)\n\nUses the projection method of PQ Axiverse paper (Appendix A) on mathcalQ to compute the locations of vacua.\n\nnote: Note\nFinding the lattice of minima when numerical minimisation is required has not yet been implemented.\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.generate.vacua_MK-Tuple{Matrix{Float64}, Matrix{Int64}}","page":"API","title":"CYAxiverse.generate.vacua_MK","text":"vacua_MK(L,Q; threshold=1e-2)\n\nUses the projection method of PQ Axiverse paper (Appendix A) on mathcalQ to compute the locations of vacua.\n\nnote: Note\nFinding the lattice of minima when numerical minimisation is required has not yet been implemented.\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.generate.vacua_TB-Tuple{Int64, Int64, Int64}","page":"API","title":"CYAxiverse.generate.vacua_TB","text":"vacua_TB(h11,tri,cy)\n\nCompute the number of vacua given a geometry from the KS database.\n\nFor small systems (Nax<=50) the algorithm computes the ratio of volumes of the fundamental domain of the leading potential and the full potential.\n\nFor larger systems, the algorithm only computes the volume of the fundamental domain of the leading potential. #Examples\n\njulia> using CYAxiverse\njulia> h11,tri,cy = 10,20,1;\njulia> vacua_data = CYAxiverse.generate.vacua_TB(h11,tri,cy)\nDict{String, Any} with 3 entries:\n  \"θ∥\"     => Rational[1//1 0//1 … 0//1 0//1; 0//1 1//1 … 0//1 0//1; … ; 0//1 0//1 … 1//1 0//1; 0//1 0//1 … 0//1 1//1]\n  \"vacua\"  => 11552.0\n  \"Qtilde\" => [0 0 … 0 1; 0 0 … 0 0; … ; 1 1 … -1 -1; 0 0 … 0 0]\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.generate.vacua_TB-Tuple{Matrix{Float64}, Matrix{Int64}}","page":"API","title":"CYAxiverse.generate.vacua_TB","text":"vacua_TB(L,Q)\n\nCompute the number of vacua given an instanton charge matrix Q and 2-column matrix of instanton scales L (in the form [sign; exponent])\n\nFor small systems (Nax<=50) the algorithm computes the ratio of volumes of the fundamental domain of the leading potential and the full potential.\n\nFor larger systems, the algorithm only computes the volume of the fundamental domain of the leading potential. #Examples\n\njulia> using CYAxiverse\njulia> h11,tri,cy = 10,20,1;\njulia> pot_data = CYAxiverse.read.potential(h11,tri,cy);\njulia> vacua_data = CYAxiverse.generate.vacua_TB(pot_data[\"L\"],pot_data[\"Q\"])\nDict{String, Any} with 3 entries:\n  \"θ∥\"     => Rational[1//1 0//1 … 0//1 0//1; 0//1 1//1 … 0//1 0//1; … ; 0//1 0//1 … 1//1 0//1; 0//1 0//1 … 0//1 1//1]\n  \"vacua\"  => 11552.0\n  \"Qtilde\" => [0 0 … 0 1; 0 0 … 0 0; … ; 1 1 … -1 -1; 0 0 … 0 0]\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.generate.vacua_full-Tuple{Matrix{Float64}, Matrix{Int64}}","page":"API","title":"CYAxiverse.generate.vacua_full","text":"vacua_MK(L,Q; threshold=1e-2)\n\nNew implementation of MK's algorithm – testing!\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.generate.vacua_id-Tuple{Matrix{Float64}, Matrix{Int64}}","page":"API","title":"CYAxiverse.generate.vacua_id","text":"vacua_id(L, Q; threshold)\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.generate.vacua_id_basis-Tuple{Matrix{Float64}, Matrix{Int64}}","page":"API","title":"CYAxiverse.generate.vacua_id_basis","text":"vacua_id_basis(L,Q; threshold)\n\nCompute the number of vacua given an instanton charge matrix Q and 2-column matrix of instanton scales L (in the form [sign; exponent])  and a threshold for:\n\nfracLambda_aLambda_j\n\ni.e. is the instanton contribution large enough to affect the minima.  This function uses JLM's method outlined in [TO APPEAR].\n\n#Examples\n\njulia> using CYAxiverse\njulia> h11,tri,cy = 10,20,1;\njulia> pot_data = CYAxiverse.read.potential(h11,tri,cy);\njulia> vacua_data = CYAxiverse.generate.vacua_id_basis(pot_data[\"L\"],pot_data[\"Q\"]; threshold=0.01)\nDict{String, Any} with 3 entries:\n  \"θ∥\"     => Rational[1//1 0//1 … 0//1 0//1; 0//1 1//1 … 0//1 0//1; … ; 0//1 0//1 … 1//1 0//1; 0//1 0//1 … 0//1 1//1]\n  \"vacua\"  => 11552.0\n  \"Qtilde\" => [0 0 … 0 1; 0 0 … 0 0; … ; 1 1 … -1 -1; 0 0 … 0 0]\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.minimizer","page":"API","title":"CYAxiverse.minimizer","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"CYAxiverse.minimizer","category":"page"},{"location":"api/#CYAxiverse.minimizer","page":"API","title":"CYAxiverse.minimizer","text":"CYAxiverse.minimizer\n\nSome minimization / optimization routines to find vacua and other such explorations.\n\n\n\n\n\n","category":"module"},{"location":"api/","page":"API","title":"API","text":"Modules = [CYAxiverse.minimizer]\nPages = [\"minimizer.jl\"]","category":"page"},{"location":"api/#CYAxiverse.minimizer.grad_std-Tuple{Int64, Int64, Int64, Vector, Matrix}","page":"API","title":"CYAxiverse.minimizer.grad_std","text":"grad_std(h11::Int,tri::Int,cy::Int,LV::Vector,QV::Matrix)\n\nTBW\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.minimizer.grad_std-Tuple{Vector, Matrix}","page":"API","title":"CYAxiverse.minimizer.grad_std","text":"grad_std(LV::Vector,QV::Matrix)\n\nTBW\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.minimizer.minima_lattice-Tuple{Matrix{Float64}}","page":"API","title":"CYAxiverse.minimizer.minima_lattice","text":"minima_lattice(v::Matrix{Float64})\n\nTBW\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.minimizer.minimize-Tuple{Vector, Matrix, Vector}","page":"API","title":"CYAxiverse.minimizer.minimize","text":"minimize(LV::Vector,QV::Matrix,x0::Vector)\n\nTBW\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.minimizer.subspace_minimize-Tuple{Vector, Matrix}","page":"API","title":"CYAxiverse.minimizer.subspace_minimize","text":"subspace_minimize(L::Vector, Q::Matrix; runs=10_000)\n\nMinimizes the subspace with runs iterations\n\n\n\n\n\n","category":"method"},{"location":"api/#CYAxiverse.read","page":"API","title":"CYAxiverse.read","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"CYAxiverse.read","category":"page"},{"location":"api/#CYAxiverse.read","page":"API","title":"CYAxiverse.read","text":"CYAxiverse.read\n\nFunctions that access the database.\n\n\n\n\n\n","category":"module"},{"location":"api/","page":"API","title":"API","text":"Modules = [CYAxiverse.read]\nPages = [\"read.jl\"]","category":"page"},{"location":"api/#CYAxiverse.cytools_wrapper","page":"API","title":"CYAxiverse.cytools_wrapper","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"add_functions/CYAxiverse.cytools_wrapper","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [CYAxiverse.cytools_wrapper]\nPages = [\"add_functions/cytools_wrapper.jl\"]","category":"page"},{"location":"userguide/#User-guide","page":"User guide","title":"User guide","text":"","category":"section"},{"location":"userguide/","page":"User guide","title":"User guide","text":"warning: Warning\nUnder construction","category":"page"},{"location":"#CYAxiverse.jl","page":"Home","title":"CYAxiverse.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A Julia package to compute axion/ALP spectra from string theory (using output of CYTools)","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Authors","page":"Home","title":"Authors","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Viraf M. Mehta","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"note: Note\nCurrently this package can run in a docker container with CYTools – DockerFile will be uploaded here once written","category":"page"},{"location":"","page":"Home","title":"Home","text":"warning: Warning\nThis package is currently not registered with the julia package manager and is still under development.  Use at your own risk!","category":"page"},{"location":"#Acknowledgements","page":"Home","title":"Acknowledgements","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This project was born after publication of Superradiance in String Theory and I am grateful to my collaborators for their input while this code was evolving.","category":"page"}]
}
