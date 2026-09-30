# Gate B one-modulus inverse-metric discriminator

This fixture checks the prescribed full inverse-metric reduction. It is an
analytic synthetic test and does not use a named source benchmark.

Use one Kähler modulus with

\[
  \mathcal V = c\tau^{3/2},\qquad
  s=g_s^{-1},\qquad
  \widehat\xi=\xi s^{3/2},\qquad
  Y=\mathcal V+\widehat\xi/2,
\]

and define \(x=\widehat\xi/\mathcal V\) and \(q=1+x/2\). The retained
potential contains only \(W=W_0\), with explicit heavy-sector F terms omitted
under the approved flux assumptions. Its no-scale combination is

\[
  \mathcal C = K_T (K_{\rm full}^{-1})^{T\bar T}K_{\bar T}-3,
  \qquad
  V=e^K |W_0|^2\mathcal C.
\]

The real-coordinate Hessian of \(K\) is converted to the Hermitian metric by
\(K_{A\bar B}=\tfrac14\partial_A\partial_B K\), for real coordinates
\((s,\tau)\). With

\[
Y_s=\frac{3\widehat\xi}{4s},\quad
Y_{ss}=\frac{3\widehat\xi}{8s^2},\quad
Y_\tau=\frac{3\mathcal V}{2\tau},\quad
Y_{\tau\tau}=\frac{3\mathcal V}{4\tau^2},
\]

the Hessian entries are

\[
H_{ss}=s^{-2}-2\left(\frac{Y_{ss}}Y-\frac{Y_s^2}{Y^2}\right),\quad
H_{s\tau}=2\frac{Y_sY_\tau}{Y^2},\quad
H_{\tau\tau}=-2\left(\frac{Y_{\tau\tau}}Y-\frac{Y_\tau^2}{Y^2}\right).
\]

The full retained inverse uses the Schur complement:

\[
\mathcal C_{\rm full}
 = \frac{K_\tau^2}{H_{\tau\tau}-H_{s\tau}^2/H_{ss}}-3.
\]

Freezing the \(T\bar T\) block instead gives

\[
\mathcal C_{\rm frozen}
 = \frac{K_\tau^2}{H_{\tau\tau}}-3.
\]

Substitution of \(K_\tau=-2Y_\tau/Y\) gives the exact expressions used by
the regression:

\[
\mathcal C_{\rm full}
 = \frac{3x(1+7x+x^2)}{(1-x)(2+x)^2},\qquad
\mathcal C_{\rm frozen}
 = \frac{3x}{4-x},
\]

\[
\mathcal C_{\rm full}-\mathcal C_{\rm frozen}
 = \frac{81x^2}{(4-x)(1-x)(2+x)^2}.
\]

The test compares both coefficients computed from the implemented metric with
these expressions, and checks the resulting \(W_0\)-only potential. It also
requires the frozen-block coefficient to differ from the full result. No
expansion in \(x\) is used. The normalization is the stated
\(K=K_{cs}-\ln(2s)-2\ln Y\), \(K_{A\bar B}=H_{AB}/4\), and
\(V=e^K|W_0|^2\mathcal C\).
