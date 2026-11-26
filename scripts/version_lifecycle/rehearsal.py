"""Closed, read-only-gated CYAx 0125 rehearsal route driver.

This module contains the approved request, result, approval-marker, and
operation-table contracts as immutable embedded bytes. It never accepts an
API path, method, command, or case order from a request.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import http.client
import json
import re
import ssl
import time
import zlib
from typing import Any, Callable, Mapping, Protocol

from .codec import sha256_hex
from .github_protection import (
    ApiResponse,
    GitHubProtectionError,
    RehearsalTriggerRequest,
    _FULL_REF_SHA,
    _REQUEST_ID,
    _git_read,
    _request_object_pairs,
)


REPOSITORY = "Julia-meets-String-Theory/CYAxiverse.jl"
REPOSITORY_ID = 527718962
OWNER_ID = 102535039
APP_ID = 5371554
INSTALLATION_ID = 172789630
APP_LOGIN = "cyaxiverse-lifecycle[bot]"
ENVIRONMENT_ID = 24648183439
ENVIRONMENT_NAME = "lifecycle"
APPROVAL_MARKER_PREFIX = "CYAX125_APPROVAL_V1:"
APPROVAL_API_VERSION = "2022-11-28"
MAX_PROVIDER_BODY = 1024 * 1024
MAX_WINDOW_SECONDS = 30.0

# These four artifacts are adopted immutable source contracts. The constants
# below are populated with exact base85 encodings of their reviewed bytes.
_EMBEDDED_CONTRACTS_B85 = {
    'request_schema': (
        'dkP>RB6N9hWg<EtB5z`9Wn*+AED9hXB5-'
        'nVaAk6IX=QUFIv{%rARr(hB6DMCWo=<!c4cyNX>V>KIv{%rARr(hARr=RZ*FsRB03;43LqdLAbl(fARr(hB64MMb!BsOUuk3'
        '_Iv{%rARr(hARr=id2nSSIv^r*baH8KXCf>LARr(hARr=eVRUq5a&96zAR=B{K`mNgEqX95IbB;pEm~nMdN3_HE?+HOdoV0E'
        'GJRV?Em~nMdN3_HT_hq3ARr(heJlzfARr(jX>D+9Wo>0{bYXO9Z*E_6Xkj8cAbScRARr(hAR=^maAhJoAR=>ga%pa7A}k6ZA'
        'Rr(hAR=&KbaZ8MZX!A$B3@fCEjeK=W?g$UFnuH<3LqdLAbl(fARr(hB6n|cYi4Y3cVA*`Z(?6_Xkj8cAbScRARr(hAR=^maA'
        'hJoAR=>ga%pa7A}k6ZARr(hAR=&KbaZ8MZX!A$B3@fCEjeK=W?g$UFnuH<3LqdLAbl(fARr(hB4v1RWn*+@WM6h|ZC`U}VIn'
        '#pdkP>RARr(hB6N9hWg<EtB6D<dX>MmCED9hXARr(hB5+}JbY*gGB03-'
        '<URy9NIbkhkU3)YzeIz0ZARr(heJlzfARr(jV_|G;Wn^DtZ)9a(Vr*|>b0Rt*dkP>RARr(hB6N9hWg<EtB5z`9Wn*+AED9hX'
        'ARr(hB5i4IP;zf@WpZ?BWpg4rATc&93LqdLARr(jVPs@!bZKvHVQf%xZ*XODbZKRCB03;@3LqdLARr(hAR=^maAhJoAR=>ga'
        '%pa7A}k6ZARr(hARr(jaA9<GWpZvJIv^rmTQDs-VJ&7|do(b8Bq9nRARr(hAbkoTARr)pED9hXARr=eb#icTb7dkrAbScRAR'
        'r(hAR=RLZgX@ZIv^rvVRU6JVJ&iRb#!GdaB^>NW+Dn8ARr)pED9hXARr=eY+-'
        'I9Iv{%rARr(hARr=id2nSSIv^r%Vrpe$bRsMYARr(hARr=ea&K^Da&&2Bb0Rt*dkP>RARr(hARr=eY+-'
        'I+c4cyNX>V>KIv{%rARr(hARr(hARr=RZ*FsRB03;33LqdLARr(hAbl(fARr(hARr(hB64MLZ*ysMZ*qAeIv{%rARr(hARr('
        'hARr=RZ*FsRB03-'
        '<N_A{$VJ&TCWpr~bQ*?4^Zf7l2Xk~A5c`rj*L3n9)WpZ<6E^2Hd3LqdLARr(hAbl(fARr(hARr(hB4c50WNBn!bY)+2Xkj8c'
        'AbScRARr(hARr(hAR=^maAhJoAR=>ga%pa7A}k6ZARr(hARr(hAR=&KbaZ8MZX!A$B3@fCEjeK=W?g$UFnuH<3LqdLARr(hA'
        'bl(fARr(hARr(hB4c50WNBn!bY)+3a%E*AIv{%rARr(hARr(hARr=id2nSSIv^r*baH8KXCf>LARr(hARr(hARr=eVRUq5a&'
        '96zAR=B{FfBP@EoNPNG%$T6A_^cNARr(hARv7#3LqdLARr(hAR>2fa%*O6Z+Bl}Y;R&;b7)~AIv{%rARr(hARr(hARr=id2n'
        'SSIv^r*baH8KXCf>LARr(hARr(hARr=eVRUq5a&96zAR=B{FfBP@EoNPNG%$T6A_^cNARr(hARv7#3LqdLARr(hAR=R7Y;0v'
        '@Ut@1%WnW@!Z(?&IIv{%rARr(hARr(hARr=id2nSSIv^r%Vrpe$bRsMYARr(hARr(hARr=bX>L$*Z*XODbZKRCB03;3HY^Gt'
        'ARr(hARr(hAR=L8WNCD1Z*F01P;zf@WpZ?BWpg4rAbScRARr(hARr(hARr(jba`-PB03-'
        '<b98cPZf7Da3LqdLARr(hARr(hAR=&KbaZ8MZX!A$B3@fCEjeK=W?g$UFnuH<3LqdLARr(hARr)p3LqdLARr(hAbl(fARr(h'
        'ARr(hB4%lLbairNUt(c%Wg<EtdkP>RARr(hARr(hB4clEb95p)AR=ZlGG=08IAvosGG#PlH#IRZH8wP6HZU?_F*Rl~HZeCdW'
        '@R@wGG!tPARr(hARr(heJlzfARr(hARr(jW@&hIb#i52W@aKfAbScRARr(hARr(hAR=RLZgX@ZIv^rpHe_ZuV>dWuGB;&qI5'
        '9UjF=JyeHexV0I5S~2FgRsoV_{`vGcqtD3LqdLARr(hAbl(fARr(hARr(hB4%lLbairNUw3bEYi4Y3cVBQ}bZ8<vAbScRARr'
        '(hARr(hAR=RLZgX@ZIv^r5GGS&oIWjpmIW;w5WHvZ7H8D43H(@n5V`VjDIAJt1F=jPkIX5vP3LqdLARr(hAbl(fARr(hARr('
        'hB6DSQb#Py4ZgypEbZ>HbUvp?-'
        'GBq|LIv{%rARr(hARr(hARr=id2nSSIv^r*baH8KXCf>LARr(hARr(hARr=eVRUq5a&96zAR=B{FfBP@EoNPNHZ*-'
        'CA_^cNARr(hARv7#3LqdLARr(hAR=XMc4=~NZf#|5bYEq7aBp&SUvp?-'
        'GBq|LIv{%rARr(hARr(hARr=id2nSSIv^r*baH8KXCf>LARr(hARr(hARr=eVRUq5a&96zAR=B{FfBP@EoNPNHZ*-'
        'CA_^cNARr(hARv7#3LqdLARr(hAR=>RWn^D+Wn*P&aCCEDb7)~QH8vtTAbScRARr(hARr(hAR=^maAhJoAR=>ga%pa7A}k6Z'
        'ARr(hARr(hAR=&KbaZ8MZX!A$B3@fCEjeK=W?g$WG<_r@3LqdLARr(hAbl(fARr(hARr(hB5!bIa$$67Z*E_7VPb4$Uvp?-'
        'GBq|LIv{%rARr(hARr(hARr=id2nSSIv^r*baH8KXCf>LARr(hARr(hARr=eVRUq5a&96zAR=B{FfBP@EoNPNHZ*-'
        'CA_^cNARr(hARv7#3LqdLARr(hAR=R7b7fy|a%5$4B03;@3LqdLARr(hARr(jV{dMAbRs$+TM8f`ARr(hARr(hARr=8GA&_n'
        'a4lnUWnpw>A}k6ZARr(hARr(hARr(jP%<rHaBwYkaAaY0Wg;vJARr(hARr(hARr(hA~azlED9hXARr(hARr(hARr<%Vl8fFW'
        '+E&KARr(hARr(hARr(hA~a$xWMyn+bY&te3LqdLARr(hARr(hAR;tlA}k6ZARr(hARr(hARr(jG-'
        'NGea$#;`Xf0!MWnpw>A}k6ZARr(hARr(hARr(jG-NGea$#;`Xf1VcWMOn=A}k6ZARr(hARr(hARr(jG-NGea$#;`Xf0%AY-'
        'Mz1A}k6ZARr(hARr(hARr(jG-NGwVP`F4a%Ew3Wg;vJARr(hARr(hARr(hA~a+zbYW*Lb#P>1bY&te3LqdLARr(hARr(hAR;'
        'tmEp%aLEo5bEWprgCED9hXARr(hARr(hARr<%Wg;vJARr(hARr(hARr(hA~a?#V{&C-'
        'bY&te3LqdLARr(hARr(hAR;toEp2ahWg;vJARr(hARr(hARr(hB2YCgWp-t5bS-aUb7gXNWg;vJARr(hARr(hARr(hB2YCgW'
        'OZ<CX=7n@Wi4i9ZewL4ED9hXARr(hARr(hARr<%XCew9ARr(hARr(hAYBR|ARr(hARr)pED9hXARr(hARr=dcWz~JUtwc(Z*'
        'pI0WFk5sdkP>RARr(hARr(hB4clEb95p)ATcm9H8V9ZGdT(%ARr(hARr)pED9hXARr(hARr=PaByE~WFk5sdkP>RARr(hARr'
        '(hB4clEb95p)AT={LF*P+b3LqdLARr(hAbl(fARr(hARr(hB57`ObYX04VRUJ4ZeM9+B03;@3LqdLARr(hARr(jV{dMAbRs$'
        '+F*h<dI5{>mFbW_bARr(hARv7#3LqdLARr(hAR=jQbY*UIUuk3_Iv{%rARr(hARr(hARr=id2nSSIv^r*baH8KXCf>LARr(h'
        'ARr(hARr=bX>LqqZfA68B03;33LqdLARr(hAbkoTARr(hAbl(fARr(hARr=gWpQ<Ba%E&9Iv`sLARr(hARr(hB5-'
        'VBZeMm~a&u{KZXzrSARr(hARr(hB64MLZ*ysMZ*qAeED9hXARr(hARr=RVQyq;WMOn=Uvp?-'
        'A}k6ZARr(hARr(jV_|M&X=Gt^WnXl1Wo05P3LqdLARr(hAR>2fa%*O6Z+Bl}Y;R&;b7)~AED9hXARr(hARr=RVQg$=WM5-'
        '%WMyAsY;R(7A}k6ZARr(hARr(jW@&hIb#i52VqtS-'
        'A}k6ZARr(hARr(jW@&hIb#i52W@aKR3LqdLARr(hAR=aIcyx7gWnXu1a%*O6Z+BmCVRUFBED9hXARr(hARr=hWps6LUukZ3W'
        'o~qDa(Q2KXkjunHX<wvARr(hARr(hB4uuNX>xCFZDnqBUuAf3Z*p{Bb7)~QH8vtF3LqdLARr(hAR=>RWn^D+Wn*P&aCCEDb7'
        ')~QH8vtF3LqdLARr(hAR=#YWpZJ3X>V>{bYWs_WnXh>VKOx~A}k6ZARr(hARr(jV_|b;UvF|`WpW}c3LqdLARr(hAR=#fZe?'
        ';`VPkY}a$jj=A}k6ZARr(hARr(jVQ_F?X=EZS3LqdLARr(hAR=jQb97;BY+-'
        'b1Z*E^{WFjmIARr(hARr(hB57`PWo~p|X=EY_ARr(hARt{V3LqdLARr(jVPs@!bZKvHVQf%xZ*XODbZKRCB03;uVQh0{3Lqd'
        'LAbl(fARr(hB5-VBZeMd~VKOx~B03;@3LqdLARr(jba`-PB03-<b98cPZf7Da3LqdLARr(jaA9<GWpZvJIv^rmTQDs-VJ&7|'
        'dp0zEBq9nRARr)pED9hXARr=dcWz~JUtx4~Wpi|4bZKvHB03;@3LqdLARr(jba`-PB03-'
        '<Z(?d?V{{@c3LqdLARr(jaB^>OWpZ?BWpg4rAbScRARr(hARr(jVRUq5b97;JX>V>{c4cyNX>V>KIv{%rARr(hARr(hARr=R'
        'Z*FsRB03;33LqdLARr(hAbl(fARr(hARr(hB64MMb!BsOUuk3_Iv{%rARr(hARr(hARr=id2nSSIv^r*baH8KXCf>LARr(hA'
        'Rr(hARr=bX>LqqZfA68B03;33LqdLARr(hAbl(fARr(hARr(hB57`PWo~p|X=EZgAbScRARr(hARr(hAR=^maAhJoAR=>ga%'
        'pa7A}k6ZARr(hARr(hAR=vPZcJrvXLM*HIv_C$ARr(hARr(heJlzfARr(hARr(jaBN|2Uvp?-'
        'GBq|LIv{%rARr(hARr(hARr=id2nSSIv^r*baH8KXCf>LARr(hARr(hARr=eVRUq5a&96zAR=B{FfBP@EoNPNHZ*-'
        'CA_^cNARr(hARv7#3LqdLARr(hAR=R7Ze(d>VRU6*b7)~AIv{%rARr(hARr(hARr=id2nSSIv^r*baH8KXCf>LARr(hARr(h'
        'ARr=eVRUq5a&96zAR=B{FfBP@EoNPNG%$T6A_^cNARr(hARv7#3LqdLARr(hAR=XOZ)0d*X=EZgAbScRARr(hARr(hAR=^ma'
        'AhJoAR=>ga%pa7A}k6ZARr(hARr(hAR=vPZcJrvXLM*HIv_C$ARr(hARr(heJlzfARr(hARr(jW^!d^dSzd7b!=sGWprO@WF'
        'k5sdkP>RARr(hARr(hB4clEb95p)ATl;HF*Z0hHVPmhARr(hARv7#3LqdLARr(hAR=aUY;0d-'
        'WoBt^X>@6CZeMd~VKOx~B03;@3LqdLARr(hARr(jba`-PB03-<b98cPZf7Da3LqdLARr(hARr(jaA9<GWpZvJIv^rmTQDs-V'
        'J&7|dp0zEBq9nRARr(hARr)pED9hXARr(hARr=TW@cq$bZK^FUvhP9WnXDzb0Rt*dkP>RARr(hARr(hB4clEb95p)AX^F`AR'
        'r(hARr(hARr(zHZw6aFgP|W3LqdLARr(hARr(hATl;HF*GtVH!KPuARr(hARr(hARr(zHZw6cI5sv4ARr(hARr(hARt`|ARr'
        '(hARr(heJlzfARr(hARr(jZ+C8Ga$jL%bZ>HBX=EZgAbScRARr(hARr(hAR=RLZgX@ZIv_DHGBq<bFf%y{ARr(hARr(heJlz'
        'fARr(hARr(jY+-'
        'YBUvF?_a$$67Z*E_7WpZt4ZeeU+a%E#>X>fF3b7)~QH8vtTAbScRARr(hARr(hAR=^maAhJoAR=>ga%pa7A}k6ZARr(hARr('
        'hAR=&KbaZ8MZX!A$B3@fCEjeK=W?g$WG<_r@3LqdLARr(hAbl(fARr(hARr(hB5rSAaAj^}X>Mm<Z*pICZfkCDcWz&Ia%psB'
        'b0Rt*dkP>RARr(hARr(hB4clEb95p)AartdWeOl5ARr(hARv7#3LqdLARr(hAR=XWV{CPEX?A5_Z+C8Ga$k3HX>?_BUv+MDX'
        '>4C|Wn*t{V`*$@VRUJ4ZX!A$dkP>RARr(hARr(hB4clEb95p)AartdWeOl5ARr(hARv7#3LqdLARr(hAR=XOZ)0d*b97;HbY'
        ')~;VRRxoAbScRARr(hARr(hAR=^maAhJoAR=>ga%pa7A}k6ZARr(hARr(hAR=vPZcJrvXLM*HIv_C$ARr(hARr(heF`8TARr'
        '(heJlzfARr(hAR=;Qadl~OWn>~cAX^F`ARr(hARr(jVRUq5b97;JX>V>{c4cyNX>V>KED9hXARr(hARr=gWpQ<7b97&6WFjm'
        'IARr(hARr(hB57`PWo~p|X=EZS3LqdLARr(hAR=&VVQyb@XkjunHX<wvARr(hARr(hB4c50WNBn!bY)+2Xkj8O3LqdLARr(h'
        'AR=XOZ)0d*X=EZS3LqdLARr(hAR=aRWo3G0UvhP9Wpib8Uuk3_ED9hXARr(hARr=Ub!=>3WMyV)ZfSIBZ*E_6XkjunHX<wvA'
        'Rr(hARr(hB4uV~Wn*+{c4c33b!=r{X=HOEED9hXARr(hARr=dcWz~JUtwc(Z*pI0WFjmIARr(hARr(hB5YxEbYE|9WpZJ3X>'
        'V>{bY*gFX>MU`Uvgz*WodA9Uvp?-'
        'GBq|LED9hXARr(hARr=cZ(nd_Ze(e0XJ2n}Uv+M4Zf|#PUw3k8bY*iQED9hXARr(hARr=Tcw=mJb7^*EUvGDAWpZD4a%psBa'
        '$j|BbZKl~a%E$0ZewX|X<>9}Z*C$i3LqdLARr(hAR=XOZ)0d*b97;HbY)~;VRRx2ARr(hARt{V3LqdLARr(jVPs@!bZKvHVQ'
        'f%xZ*XODbZKRCB03;uVQh0{3LqdLAbl(fARr(hB5!wYWpZC(baZ8NbYXO9Z*E_6XkjunHX=G8dkP>RARr(hB6N9hWg<EtB6D'
        '<dX>MmCED9hXARr(hB5+}JbY*gGB03-'
        '<URy9NIbkhkU3)e(eIz0ZARr(heF`8TeJlzfAR=;Qadl~OWn>~cAX^F`ARr(jb7N>_ZDC(_WpZ<AZ*C$i3LqdLAR=;Qadl;L'
        'bYE#?A}k6ZARr(jX>D+9Wo>0{bYXO9Z*E_6Xkj8O3LqdLAR>2fa%*O6Z+Bl}Y;R&;b7)~AED9hXARr=TcyMK7bY)~;c5Q86b'
        '7)~AED9hXARr=RVQg$=WM5-%WMyAsY;R(7A}k6ZARr(jaCLHUZ*yfLED9hXARr=eY+-I9ED9hXARr=eY+-'
        'I+b7)~QH8vtF3LqdLAR=#fZe?;`VRUq5b97;JX>V>KED9hXARr=dcWz~JUtx4~Wpi|4bZKvHUvp?-'
        'GBq|L3Lqd|ED9hXB4K1?X>@6CZeeUta&K^Da&&2Bb0Rt*W?^h|WeR-?'
    ),
    'driver_return_schema': (
        'dkP>RB6N9hWg<EtB5z`9Wn*+AED9hXB5-'
        'nVaAk6IX=QUFIv{%rARr(hB6DMCWo=<!c4cyNX>V>KIv{%rARr(hARr=RZ*FsRB03;33LqdLAbl(fARr(hB64MMb!BsOUuk3'
        '_Iv{%rARr(hARr=id2nSSIv^r*baH8KXCf>LARr(hARr=bX>LqqZfA68B03;33LqdLAbl(fARr(hB5-'
        'VBZeMd~VKOx~B03;@3LqdLARr(jba`-PB03-<b98cPZf7Da3LqdLARr(jaA9<GWpZvJIv^rmTQDs-VJ&7|dp0zEBq9nRARr)'
        'pED9hXARr=RVRL0)X=EZgAbScRARr(hAR=XMb!{R#AX^F`ARr(hARr(jP%<rHaBwYSa%Ew3Wg;vJARr(hARr(hB2Y3dVQ_FQ'
        'b#P>1bY&te3LqdLARr(hAR;tjA}k6ZARr(hARr(jG-'
        '54oW@aKR3LqdLARr(hAR;tkEo5bEWprgCED9hXARr(hARr<%V<IdHARr(hARr(hA~a+zVsc?_V`wd7a%Ew3Wg;vJARr(hARr'
        '(hA~a+zVsc?_V`wdPaAaY0Wg;vJARr(hARr(hA~a+zVsc?_V`wd8Wo%`1Wg;vJARr(hARr(hA~a+zbYW*LV{&C-'
        'bY&te3LqdLARr(hAR;tmEp%aLEp>2YVRU68ED9hXARr(hARr<%WG!@IXDwu9Y-Mz1A}k6ZARr(hARr(jG-'
        'V<z3LqdLARr(hAR;toEn{+JVRU68ED9hXARr(hARr<%W-'
        'V=Rc4Z<g3LqdLARr(hAR<sTEoF9PZgeegVsmA3c4Z<g3LqdLARr(hAR<sTEo60YY-wX*bY(4OWo~0-A}k6ZARr(hARr(jG-'
        'o0TARr(hARt`|ARr(heJlzfARr(jb97;Jb#o#*AbScRARr(hAR=XMb!{R#AX^F`ARr(hARr(jP(f2uUqM4dMNm{lL?SE-'
        'ARr(hARr(hB2Yn7Q(r_yPDw>XUsOR-M@3X$Lr+drQcp}GED9hXARr(hARr=8K~qy-'
        'PghPwQeRR}Ra8Y^PETJ@RZ~bJED9hXARr(hARr=BR8LS}PghPwQeQzsR7p=xUs6R;RY_7sL?SE-ARr(hARr(hB0@}0LrX<OU'
        'sX;^PES`(A}k6ZARr(hARr(jLQGFXOGQLqPES-{Rz*@tMoB?JOhqCr3LqdLARr(hAR<CcPeV&ZL|;`-'
        'MOaWpLsUgXUqM4dMNm{hPD4c^ED9hXARr(hARr=6PgGw;SVcosR7FH03LqdLARr)J3LqdLAbl(fARr(hB4u`IWMyt+WnXh>V'
        'KOx~B03;@3LqdLARr(jba`-PB03-<b98cPZf7Da3LqdLARr(jaA9<GWpZvJIv^rmTQDs-VJ&7|dp0zEBq9nRARr)pED9hXAR'
        'r=cWq5R7V_|b;B03;@3LqdLARr(jWo~tCB03;j3LqdLARr(hAR<sQEn#qQEn{+JVRU68ED9hXARr(hARr=8GA&_na4mIkWMO'
        'n=A}k6ZARr(hARr(jG+`nv3LqdLARr(hAR;tkEpBFJA}k6ZARr(hARr(jG-'
        '54eWo%`1Wg;vJARr(hARr(hA~a(nED9hXARr(hARr<%WG!NHVQyn+En{+JVRU68ED9hXARr(hARr<%WG!NHVQyn+Ep>2YVRU'
        '68ED9hXARr(hARr<%WG!NHVQyn+Eo5bEWprgCED9hXARr(hARr<%WG!@IXDwrLWnpw>A}k6ZARr(hARr(jG-'
        'NGwVP`FMaAaY0Wg;vJARr(hARr(hA~a+zbYW*LWMyn+bY&te3LqdLARr(hAR;tnA}k6ZARr(hARr(jG-fSha%Ew3Wg;vJARr'
        '(hARr(hA~a?#ZEtpEA}k6ZARr(hARr(jP&F-Oc4cmKEpK9TWpZ|9A}k6ZARr(hARr(jP&F-'
        'Nb#QEHV_|e<EoNnIV`U;N3LqdLARr(hAR;tpA}k6ZARr(hARr)ab!==3ARr(hARt`|ARr(heJlzfARr(jb#8KHb8l>RWn>~c'
        'AbScRARr(hAR=^maAhJoAR=OKZ){~@ZXyaGARr)pED9hXARr=eHD7aeVsd43b!>DZIv{%rARr(hARr=dZe>qqB03;j3LqdLA'
        'Rr(hAbScRARr(hARr(hAR=^maAhJoAR=ycY-}P5ARr(hARr(heJlzfARr(hARr)n3LqdLARr(hARr(jba`-PB03-'
        '<Z(?d?V{{@c3LqdLARr(hARr(jaB^>OWpZ?BWpg4rAbScRARr(hARr(hARr(jZDDj{XlZU|Utw@?UvzS5XJ=({UuSh;a%5j+'
        'cx7^9X>(;{B03;@3LqdLARr(hARr(hARr(jV{dMAbRs$+W?^h|WeOl5ARr(hARr(hARv7#3LqdLARr(hARr(hAR=#Kb7gXNW'
        'n^D(W@cY<WoBPxc4cmKB03;@3LqdLARr(hARr(hARr(jba`-'
        'PB03;j3LqdLARr(hARr(hARr(hAR=OKZ){~@ZXzrSARr(hARr(hARr(hARr(hB5rkTY$6IEARr(hARr(hARr(hAYBR|ARr(h'
        'ARr(hARr)pED9hXARr(hARr(hARr=hWn*t{WM5=*X?A6DUu0=>aA9;~XkTM*b#8PbIv{%rARr(hARr(hARr(hARr=id2nSSI'
        'v`sLARr(hARr(hARr(hARr(hB57`PWoKn_A}k6ZARr(hARr(hARr(hARr(jZgp&IA_^cNARr(hARr(hARr(hT`US9ARr(hAR'
        'r(hARr(hAR=vPZfR|GZ6Z1#FbW_bARr(hARr(hARv7#3LqdLARr(hARr(hAR=LKX<u!1bYXO9Z*E^>Z*^{TB03;@3LqdLARr'
        '(hARr(hARr(jba`-'
        'PB03;j3LqdLARr(hARr(hARr(hAR=jQbY*8{aw04WARr(hARr(hARr(hARr(hB5rkTY$6IEARr(hARr(hARr(hAYCj9ARr(h'
        'ARr(hARr(hARr=bX>Mt4b!{R#ATSCbARr(hARr(hARr)pED9hXARr(hARr(hARr=VX>?z3b#rK6V{dhCbRs$+dkP>RARr(hA'
        'Rr(hARr(hB6N9hWg<EtTM8f`ARr(hARr(hARr(hARr=XZggd5WpW}c3LqdLARr(hARr(hARr(hAR=ycY-'
        '}P5ARr(hARr(hARr(hARt{V3LqdLARr(hARr(hARr(jZE0?4ZFOxTIv_9#ARr(hARr(hARr(heJlzfARr(hARr(hARr(jWq4'
        '_GbZKs9UuI=)V`X1+Xkj8cAbScRARr(hARr(hARr(hAR=^maAhJoAX^F`ARr(hARr(hARr(hARr(jb98cPZf7Da3LqdLARr('
        'hARr(hARr(hAR=ycY-}P5ARr(hARr(hARr(hARt{V3LqdLARr(hARr(hARr(jaA9<GWpZvJIv^rmTQDs-VJ&7|do(b8Bq9nR'
        'ARr(hARr(hARr)pED9hXARr(hARr(hARr=UWo~0-'
        'UvqR}bY&tsAbScRARr(hARr(hARr(hAR=XMb!{R#AX^F`ARr(hARr(hARr(hARr(jRz*@tMoC3PUrj+&Lr5Yl3LqdLARr(hA'
        'Rr(hARr(hAR<9RQ$<cxA}k6ZARr(hARr(hARr(hARr(jRZdGzPghPNED9hXARr(hARr(hARr(hARr=5NmEThR6|H23LqdLAR'
        'r(hARr(hARr)J3LqdLARr(hARr(hAbl(fARr(hARr(hARr(hB64MMb!BsOUvO+;ZeML-bYo~DIv{%rARr(hARr(hARr(hARr'
        '=id2nSSIv`sLARr(hARr(hARr(hARr(hB4TfEY-'
        'M3?A}k6ZARr(hARr(hARr(hARr(jZgp&IA_^cNARr(hARr(hARr(hT?!x|ARr(hARr(hARv7#3LqdLARr(hARr(hAR=aAX>4'
        '_JWnXe-VRLV8B03;@3LqdLARr(hARr(hARr(jba`-'
        'PB03;j3LqdLARr(hARr(hARr(hAR=>ga%pa7A}k6ZARr(hARr(hARr(hARr(jZgp&IA_^cNARr(hARr(hARr(hT?!x|ARr(h'
        'ARr(hARv7TARr(hARr(hARv7#3LqdLARr(hARr(jVPs@!bZKvHVQf%xZ*XODbZKRCB03;uVQh0{ED9hXARr(hARr(hB64MMb'
        '!l>CWFk5sTM8f`ARr(hARr(hARr=bVRU0?X>Mm<VQ_F?baH8DXJv9<XLVt6WM5@?WpZO_b7f>AED9hXARr(hARr(hARr=dVs'
        'mA3c4cH=Z)Rp+a%E;;Wp-'
        't5bRsMYARr(hARr(hARr(hB6DS9Z*F8?WO8YCWpZC+X>)L4bYo~=V{dhCbRsMYARr(hARr(hARr(hB4Kc8Uu|`CVRUJ4ZeL?'
        '>b#8PbED9hXARr(hARr(hARr=VX>?z3b#rK6V{dhCbRsMYARr(hARr(hARr(hB4v1Kb98BLXJ2MzZewL%b7)~AED9hXARr(h'
        'ARr(hARr=UWo~0-UvqR}bY&te3LqdLARr(hARr(hAR=;Qadl;LbYF06VQyb-VRU0?A}k6ZARr(hARr(hARr(jW?^Y;b#i52a'
        '%Ew2Z*C$AARr(hARr(hARt`|ARr(hARr(heF`8TARr(hT?!x|ARv7TARv7#3LqdNa%FLKX>w&`B03;j3LqdLAR=>PXk~3-'
        'Uv_13b7^mGA}k6ZARr(ja%FLKWpi|2X=EZS3LqdLAR=&VVQyb@XkjunHX<wvARr(hB4c55WnXDzA}k6ZARr(jb97;Jb#o#t3'
        'LqdLAR=XUX=G(?V`X1+XkjunHX<wvARr(hB5q}PbYEj(b7dkd3LqdLAR={ca%FRGY<6X2A}k6ZARr(ja5Z0Zbz*X5b9HQVA_'
        '^cNT`US9AR=L8WNCD1Z*F01P;zf@WpZ?BWpg4rAZB4~b7d?FARr=PY-'
        '~?vB03;j3LqdLAbScRARr(hAR=jIB03;@3LqdLARr(hAR=&bZ*XODbZKRCB03;@3LqdLARr(hARr(jV_|b;Uuk3_Iv{%rARr'
        '(hARr(hARr(hB4uuMZ6Z1#TM8f`ARr(hARr(hARr(hB2YCgWp-t5bS-aUb7gXNWg;vJARr(hARr(hARr(hARr=8H7#UyaBOK'
        '~VRU6JW@T<;Wg-'
        'e7ARr(hARr(hARr)J3LqdLARr(hARr)p3LqdLARr(hAbl(fARr(hARr(hB64MMb!l>CWFk5sTM8f`ARr(hARr(hB4c55WnXD'
        'zA_^cNARr(hARt`|ARr(hARv7#3LqdLARr(jbZBL6B03;@3LqdLARr(hAR=&bZ*XODbZKRCB03;@3LqdLARr(hARr(ja5Z0Z'
        'bz*X5b9HQVB03;@3LqdLARr(hARr(hAR=^maAhJoAR=#KYGq?|A}k6ZARr(hARr(hARr(jaB^>OWpZ?BWpg4rAbScRARr(hA'
        'Rr(hARr(hAR=vHbYo~~Zf9R%aByFAa%pF0WpZC<bzyR3UuAe@a${+8Wn>~cAbScRARr(hARr(hARr(hARr(jV{dMAbRs$+W?'
        '^h|WeOl5ARr(hARr(hARr(heJlzfARr(hARr(hARr(hAR=#Kb7gXNWn^D(W@cY<WoBPxc4cmKB03;@3LqdLARr(hARr(hARr'
        '(hAR=^maAhJoAX^F`ARr(hARr(hARr(hARr(hAR=OKZ){~@ZXzrSARr(hARr(hARr(hARr(hARr=cb!==R3LqdLARr(hARr('
        'hARr(hAYBR|ARr(hARr(hARr(hAbl(fARr(hARr(hARr(hARr=hWn*t{WM5=*X?A6DUu0=>aA9;~XkTM*b#8PbIv{%rARr(h'
        'ARr(hARr(hARr(hB6N9hWg<EtTM8f`ARr(hARr(hARr(hARr(hB57`PWoKn_A}k6ZARr(hARr(hARr(hARr(hAR=ycY-'
        '}P5ARr(hARr(hARr(hARr(hT`US9ARr(hARr(hARr(hARr(jZE0?4ZFOxTIv_9#ARr(hARr(hARr(hARv7#3LqdLARr(hARr'
        '(hARr(jVQ^_*ZFO{EbZKvHUt@1|Zge6#AbScRARr(hARr(hARr(hARr(jba`-'
        'PB03;j3LqdLARr(hARr(hARr(hARr(jX>N37XJv9CED9hXARr(hARr(hARr(hARr(hB5rkTY$6IEARr(hARr(hARr(hARr)J'
        'ED9hXARr(hARr(hARr(hARr=bX>Mt4b!{R#ATSCbARr(hARr(hARr(hAbl(fARr(hARr(hARr(hARr=VX>?z3b#rK6V{dhCb'
        'Rs$+dkP>RARr(hARr(hARr(hARr=id2nSSIv`sLARr(hARr(hARr(hARr(hARr=XZggd5WpW}c3LqdLARr(hARr(hARr(hAR'
        'r(jZgp&IA_^cNARr(hARr(hARr(hARt{V3LqdLARr(hARr(hARr(hAR=vPZfR|GZ6Z1#FbW_bARr(hARr(hARr(heJlzfARr'
        '(hARr(hARr(hAR=XWX>)XGZf9R+Wo~0-Uvp?-'
        'B03;@3LqdLARr(hARr(hARr(hAR=^maAhJoAX^F`ARr(hARr(hARr(hARr(hAR=>ga%pa7A}k6ZARr(hARr(hARr(hARr(hA'
        'R=ycY-}P5ARr(hARr(hARr(hARr(hT`US9ARr(hARr(hARr(hARr(jaA9<GWpZvJIv^rmTQDs-VJ&7|do(b8Bq9nRARr(hAR'
        'r(hARr(hAbl(fARr(hARr(hARr(hARr=UWo~0-'
        'UvqR}bY&tsAbScRARr(hARr(hARr(hARr(jWo~tCB03;j3LqdLARr(hARr(hARr(hARr(jRz*@tMoC3PUrj+&Lr5Yl3LqdLA'
        'Rr(hARr(hARr(hARr(jK|)hSPE;Z+3LqdLARr(hARr(hARr(hARr(jRZdGzPghPNED9hXARr(hARr(hARr(hARr(hB27tCO+'
        'i#cNFoX#ARr(hARr(hARr(hARr)J3LqdLARr(hARr(hARr)pED9hXARr(hARr(hARr(hB64MMb!BsOUvO+;ZeML-bYo~DIv{'
        '%rARr(hARr(hARr(hARr(hB6N9hWg<EtTM8f`ARr(hARr(hARr(hARr(hB4TfEY-'
        'M3?A}k6ZARr(hARr(hARr(hARr(hAR=ycY-'
        '}P5ARr(hARr(hARr(hARr(hT?!x|ARr(hARr(hARr(heJlzfARr(hARr(hARr(hAR=aAX>4_JWnXe-VRLV8B03;@3LqdLARr'
        '(hARr(hARr(hAR=^maAhJoAX^F`ARr(hARr(hARr(hARr(hAR=>ga%pa7A}k6ZARr(hARr(hARr(hARr(hAR=ycY-'
        '}P5ARr(hARr(hARr(hARr(hT?!x|ARr(hARr(hARr(heF`8TARr(hARr(hARv7#3LqdLARr(hARr(hAR=L8WNCD1Z*F01P;z'
        'f@WpZ?BWpg4rAZB4~b7d?FARr(hARr(hARr(hB64MMb!l>CWFk5sTM8f`ARr(hARr(hARr(hB5h%GV`yn^XJ27(a9?zCX=i6'
        ';a$je4VRB?&Wq4(BV`+0`WFjmIARr(hARr(hARr(hARr=dVsmA3c4cH=Z)Rp+a%E;;Wp-'
        't5bRsMYARr(hARr(hARr(hARr=hWn*t{WM5=*X?A6DUu0=>aA9;~XkTM*b#8PbED9hXARr(hARr(hARr(hB4Kc8Uu|`CVRUJ'
        '4ZeL?>b#8PbED9hXARr(hARr(hARr(hB4=rIUvPDEXkTM*b#8PbED9hXARr(hARr(hARr(hB4v1Kb98BLXJ2MzZewL%b7)~A'
        'ED9hXARr(hARr(hARr(hB4%Z7V`X1+bYXO5A}k6ZARr(hARr(hARr(hAR=;Qadl;LbYF06VQyb-'
        'VRU0?A}k6ZARr(hARr(hARr(hAR=aAX>4_JWnXe-VRLV8A_^cNARr(hARr(hARt`|ARr(hARr(hARv7TARr(hARr(heF`8TA'
        'Rr(heJlzfARr(hAR=XKb7dkrAbScRARr(hARr(jaB^>OWpZ?BWpg4rAbScRARr(hARr(hAR=%zUvqV0a%FRMY;+<zAbScRAR'
        'r(hARr(hARr(jba`-PB03-'
        '<Zgp&IA_^cNARr(hARr(heF`8TARr(hARv7TARr(hARv7TARr(heJlzfARr)n3LqdLARr(jX=WliAbScRARr(hARr(jaB^>O'
        'WpZ?BWpg4rAbScRARr(hARr(hAR=>gVRUtKB03;@3LqdLARr(hARr(hAR=XMb!{R#AX^F`ARr(hARr(hARr(hAR<CcPeV&ZL'
        '|;`-'
        'OHNN$P9iJ{ARr(hARr(hARr(hARr<_Oix2gMMPgtPgGx4MN&ydNkKwPMItN;ARr(hARr(hARr(hARr<_Oix2gMMPg!PDNNyM'
        'MG3YL|;KeLq$+jK~6(OA}k6ZARr(hARr(hARr(hAR<##Pf%Y^S58GzUqM4uNl#8+Qbkc!Nm4~bA}k6ZARr(hARr(hARr(hAR'
        '<mrR9{6{MMG6oMMNSBARr(hARr(hARr(hT?!x|ARr(hARr(heF`8TARr(hARv7#3LqdLARr(hAR=;Qadl~OWn>~cAX^F`ARr'
        '(hARr(hAR=>gVRUtKA_^cNARr(hARt`|ARr(hARv7#3LqdLARr(jbZBL6B03;@3LqdLARr(hAR=&bZ*XODbZKRCB03;@3Lqd'
        'LARr(hARr(jZe@6MUt?i&Wg<EtdkP>RARr(hARr(hARr=id2nSSIv^r$b!==R3LqdLARr(hARr)pED9hXARr(hARr(hB6V(Z'
        'Wpi(Ac4cHDIv{%rARr(hARr(hARr(hB4clEb95p)AartdWeOl5ARr(hARr(heF`8TARr(hARv7TARr(hARv7TARr(heJlzfA'
        'Rr)n3LqdLARr(jX=WliAbScRARr(hARr(jaB^>OWpZ?BWpg4rAbScRARr(hARr(hAR=R7b7fy?WFk5sdkP>RARr(hARr(hAR'
        'r=RZ*FsRB03-<P&F-Nb#QEHV_|e<EoNnIV`U-'
        '=ARr(hARr(hARv7#3LqdLARr(hARr(jb97;Jb#o#*AbScRARr(hARr(hARr(jV{dMAbRs$+B2Yn7Q(r_yPDw>XUsOR-'
        'M@3X$Lr+drQcp}G3LqdLARr(hARr)p3LqdLARr(hAbl(fARr(hARr(hB64MMb!l>CWFk5sTM8f`ARr(hARr(hB4c55WnXDzA'
        '}k6ZARr(hARr(hAR=>gVRUtKA_^cNARr(hARt`|ARr(hARv7#3LqdLARr(jbZBL6B03;@3LqdLARr(hAR=&bZ*XODbZKRCB0'
        '3;@3LqdLARr(hARr(jb#8KHb8l>RWn>~cAbScRARr(hARr(hARr(jV{dMAbRs$+W?^h|WeOl5ARr(hARr(heJlzfARr(hARr'
        '(hAR=yMcywQ5VRL07Iv{%rARr(hARr(hARr(hB4clEb95p)AR;tpA_^cNARr(hARr(heJlzfARr(hARr(hAR=%zUvqV0a%FR'
        'MY;+<zAbScRARr(hARr(hARr(jaB^>OWpZ?BWpg4rAbScRARr(hARr(hARr(hAR=vHbYo~~Zf9R%aByFAa%pF0WpZC<bzyR3'
        'UuAe@a${+8Wn>~cAbScRARr(hARr(hARr(hARr(jV{dMAbRs$+W?^h|WeOl5ARr(hARr(hARr(heJlzfARr(hARr(hARr(hA'
        'R=>RV{dL`Uu1G=c4cy3WNCA7VRU0?Ut@1|Zge6#AbScRARr(hARr(hARr(hARr(jba`-PB03-'
        '<X>N37XJv9CED9hXARr(hARr(hARr(hARr=RZ*FsRB03;23LqdLARr(hARr(hARr)pED9hXARr(hARr(hARr(hB4Kc8Uu|`C'
        'VRUJ4ZeL?>b#8PbIv{%rARr(hARr(hARr(hARr(hB6N9hWg<EtB57`PWoKn_A}k6ZARr(hARr(hARr(hARr(jV{dMAbRs$+F'
        'bW_bARr(hARr(hARr(heJlzfARr(hARr(hARr(hAR=dJbYF0Fb7)^<Z*^{TB03;@3LqdLARr(hARr(hARr(hAR=^maAhJoAR'
        '=jQbY*8{aw04WARr(hARr(hARr(hARr(hB4clEb95p)ATSCbARr(hARr(hARr(hAbl(fARr(hARr(hARr(hARr=TcxiKVX>M'
        'm<W@T<;WnXh>VIn#pdkP>RARr(hARr(hARr(hARr=id2nSSIv^r*baH8KXCf>LARr(hARr(hARr(hARr(hB5+}JbY*gGB03-'
        '<URy9NIbkhkU3)YzeIz0ZARr(hARr(hARr(hARv7#3LqdLARr(hARr(hARr(jW@T<;WnXi2VRU68Iv{%rARr(hARr(hARr(h'
        'ARr(hB4clEb95p)AR<;pQb|ThMMPgsK~zIXA_^cNARr(hARr(hARr(heJlzfARr(hARr(hARr(hAR=;Qadl;LbYF06VQyb-V'
        'RU0?B03;@3LqdLARr(hARr(hARr(hAR=RLZgX@ZIv{j%b!7@5ARr(hARr(hARr(hAbl(fARr(hARr(hARr(hARr=UVQFl2a%'
        'Ep~Wnpt~ZX!A$dkP>RARr(hARr(hARr(hARr=id2nSSIv^r$b!==R3LqdLARr(hARr(hARr)p3LqdLARr(hARr(hAbkoTARr'
        '(hARr(hAbkoTARr(hARr)p3LqdLARr)p3LqdLAbl(fARr(hdkP>RARr(hB57tKIv{%rARr(hARr(hB5-'
        'nVaAk6IX=QUFIv{%rARr(hARr(hARr=RVRL0)X=EZgAbScRARr(hARr(hARr(jV{dMAbRs$+B2YCgWp-t5bS-aUb7gXNWg-'
        'e7ARr(hARr(hAbl(fARr(hARr(hARr=hbYXOLb0Rt*dkP>RARr(hARr(hARr=RZ*FsRB03-'
        '<P(f2uUqM4dMNm{lL?Q|xARr(hARr(hAbkoTARr(hARr)pED9hXARr(hARr=gWpQ<Ba%E&9Iv`sLARr(hARr(hARr=RVRL0)'
        'X=EZS3LqdLARr(hARr(jb97;Jb#o#LARr(hARr(hT?!x|ARr(heJlzfARr(hAR=^VWo{xmAbScRARr(hARr(jaB^>OWpZ?BW'
        'pg4rAbScRARr(hARr(hAR={ca%FRGY<6X2B03;@3LqdLARr(hARr(hAR=RLZgX@ZIv{3YY;$D_ARr(hARr(hARv7#3LqdLAR'
        'r(hARr(jZe@6MUt?i&Wg<EtdkP>RARr(hARr(hARr=RZ*FsRB03-<P&F-Nb#QEHV_|e<EoNnIV`U-'
        '=ARr(hARr(hARv7#3LqdLARr(hARr(ja5Z0Zbz*X5b9HQVB03;@3LqdLARr(hARr(hAR=&bZ*XODbZKRCB03;@3LqdLARr(h'
        'ARr(hARr(jb7f<1Ze(9%a%pyDa$jU=b8umFV`yJvZ*^{TB03;@3LqdLARr(hARr(hARr(hAR=^maAhJoAR=jQbY*8{aw04WA'
        'Rr(hARr(hARr(hARr(hB4clEb95p)ATSCbARr(hARr(hARr(hAbl(fARr(hARr(hARr(hARr=PaA{v{b#!5LX>V>{V{dhCbR'
        's$+dkP>RARr(hARr(hARr(hARr=id2nSSIv^rxZggd5WpW}c3LqdLARr(hARr(hARr(hAR=RLZgX@ZIv_9#ARr(hARr(hARr'
        '(hARv7#3LqdLARr(hARr(hARr(jXK8d_aCLKNUt@1|Zge6#AbScRARr(hARr(hARr(hARr(jba`-PB03-'
        '<X>N37XJv9CED9hXARr(hARr(hARr(hARr=RZ*FsRB03;23LqdLARr(hARr(hARr)pED9hXARr(hARr(hARr(hB4%M}Y;|&F'
        'Uvgz(b8l`UIv{%rARr(hARr(hARr(hARr(hB6N9hWg<EtB5rkTY$6IEARr(hARr(hARr(hAbkoTARr(hARr(hARr)p3LqdLA'
        'Rr(hARr)p3LqdLARr(hAbkoTARr(hAbkoTARr)p3Lqd|3VjL'
    ),
    'operation_table': (
        'TM8f`dkP>RARr=RVRL07Iv^rYGA&_na4lnUWnpw>A}k6ZARr(jVPs)&bY*g1Wo~qGc_KO>B5-'
        'nVVr5@+a%pF0WpZC*a%Ew3WnXh;a&~2MA}k6ZARr(ja%E;BIv^r)WoC0PXk}q!b1!mbXk}q?b75>RbaH8DXJv9PV|ihCF)}q'
        'Ra561naBwYSa%Ew3Wg;vJARr(hB4v1RWn*+@WM6M=WFk5sZgp&IED9hXARr=iVRC0>bRs$+B4#l%W@2JEWn(omWi(?qH8C(X'
        'HZ*27Ffw5=HD)q4F*h@2Wj8l6Wg;vJARr(hB4v1RWn*+@WM6V+b9HQVB03-<L`6<nUsOR-'
        'M@3X$QdLYvA}k6ZARr(jbYXI5WprO-'
        'Z*FvQZ)_quAR;n0GchzUI5i++a%Ew3X>V>KED9hXARr=hWo2Yvb98TTB03;$b!==b3LqdLAR=&bWn*t{WNCD1Z*FrUIv^rLP'
        'fblvPG3+`MLQs2Vsm9~baG{8EFgAaY-wa=cxiKVX>Mm@Z*6U9bSxlYV{~<4Y(a2vX>Mw7Vsvk7Wo|nlWq5FGX=7=0Y-'
        'x67P%?94Z*XNIED9hXARr=eZ*z2GZ*F91bZKvHb0Rt*B12D2O;1i=P)}1-J0M|nbaH88b#!53Y-M9|Wnpw_Z*FpRY-'
        'MC+ZfRj`D{^IKVPbP-Zge{!Zf`+waCCBMXJ=({YHwm?cyMK7bY)~B3LqeTED9hXdkP>RARr=RVRL07Iv^rYGA&_na4mIkWMO'
        'n=A}k6ZARr(jVPs)&bY*g1Wo~qGc_KO>B5-'
        'nVVr5@+a%pF0WpZD2aAaY0WnXh;a&~2MA}k6ZARr(ja%E;BIv^r)WoC0PXk}q!b1!mbXk}q?b75>RbaH8DXJv9PV|ihCF)}q'
        'Ra561naBwYkaAaY0Wg;vJARr(hB4v1RWn*+@WM6M=WFk5sB4#l%W@2JEWn(omWi(?qH8C(XHZ*27Ffw5=HD)q4F*h@2Wj8l6'
        'Wg;vJARr(hB6MMLXJvFEIv^rpHe_ZuV>dWuGB;&qI59UjF=JyeHexV0I5S~2FgRsoV_{`vGcqtDED9hXARr=TcyMK7bY)~;a'
        '%FRMY;+<zAR<IXPFY`6K~hIWR9{k6OhqCr3LqdLAR=^Oa%W|9Ut@1>baHQOB03-<GBz_YG%z?dAa!tLVRU66b8uy2X=Z6-'
        'VQg%9J0M0za%FLKWpi{OW^Zz1WjbbIY;$EIED9hXARr=hWo2Yvb98TTB03-'
        '<O+ijXK~zstSx<LvWpZ<6Wn^V|X>)XGZfA9HWMOn=AaZ49AYpW3VRL05cWHEJAaZ46Z*pX1WNvS6Eo^CKWn+0`Y-'
        'Mm^bZ9SbZ)$I1AaHVJV{dL`X>@6CZY&^kWpH6~VRU6rcWz~JVPkY@Z*F0AbZBpKX?kIFX>V>jAYo#2Wo~pmK2uarP+w11PDN'
        '5*K|@qYPflM_MNw5rQbj}}ED9hXARr=ea%E$0Ze(e6X>V?GB03-<Lr+ajPflM@Qbju;Wq4s@baQ28WMW}+WiN7NVPs-qV{0H'
        '`Ze$>KVQgt+Mn+_1b7N(0WMOV}J0M|Wbai2DL2z(sZfb90bZ={AZXzrSARr(hB5-'
        'eWbYpLBWNCD1Z*FrUIv^rLPfblvPG3+@Q&c-3b#P>1bY*gNY-M3|baH88b#!53Y-'
        'MC+ZfRj`D|K#TXkl(=Wn^Mub7dk5ARv7#3LqeR3LqdLAR=R7b7dkrAR;tjA}k6ZARr(jVPs)&bY*g1Wo~qGc_KO>B4c55UvP'
        '41ZewY1VQgP!W+E&KARr(hB64MBB03-'
        '<a%E<7FKA_9WOFZaWoTt#a&uv9FK}{cZewY1VQepBd0}`lGBqtSFfukVF)%SLa&L8XWg;vJARr(hB4v1RWn*+@WM6M=WFk5s'
        'B4#l%W@2JEWn(omWi(?qH8C(XHZ*27Ffw5=HD)q4F*h@2Wj8l6Wg;vJARr(hB6MMLXJvFEIv^rpHe_ZuV>dWuGB;&qI59UjF'
        '=JyeHexV0I5S~2FgRsoV_{`vGcqtDED9hXARr=TcyMK7bY)~;a%FRMY;+<zAR<9SLq$+jUqx6!LsVZzMj|W<ARr(hB6MMLXJ'
        'vF>V{dMBa&K%RIv^r4HZw6cI5sw6V{~bDWoB|^WqM^GD>61SF*GtVH*#fmX=QgoaByOIaA9+EAS*I9GchzUI5uf+bY*99X>@'
        'raED9hXARr=hWo2Yvb98TTB03-'
        '<Pj_x*a&u*6WN>n6ZewY1VQgYyb7dfMWpQ<Ba%E&7Vr6D;a%E(4X?A6DA}k6ZARr(jaB^j1Z*F91bZKvHb0Rt*B12D2O;1i='
        'P*O!8D<EZfVPkY}Vrpe$bY*yHb98erba`-PFKlUUWnpJ!FKcdZcW!cJW^;65bY&nbAZ2)AV{~I-'
        'b7f&{Y;SjLX>)WtAZ~AAa$;$8a$$0LRZ>iIFJW?Lb|NeaARr(hB5-'
        'eWbYpLBWNCD1Z*FrUIv^rLPfblvPG3+@Q&b=;AZ2)AV{~<HV`yP+XJuqBVPbP-Zge1Tav)`RVPkY|Wp`6ZK_GHvVPs-qV{0H'
        'QAYpWLa%p09bYWs_Wprh7ZE0>{Y;SdRV{dI`A_^cNeJlzfAbScRARr(jV_|b;B03-<G-'
        '54oW@aKR3LqdLAR=L8VQ_S1a$jX`baHtjIv^r&a&KZ~UvP41ZewY1VQgP+W@cY=WpZ|9aw04WARr(hB64MBB03-'
        '<a%E<7FKA_9WOFZaWoTt#a&uv9FK}{cZewY1VQepBd0}`lGBqtSFfukVF)%SLa&L8XWg;vJARr(hB4v1RWn*+@WM6M=WFk5s'
        'B4IXUW;bIuIAt<7Wo0-'
        'qH#ae3V=y*iFgG|eVKp!~Wn^PvWn?olFd{4pARr(hB6MMLXJvFEIv^ruF*0UiVmM`EH8N#1V>dN1Ff}$bW;QS~VKFskGBz<c'
        'GiGHsH!@`+ED9hXARr=TcyMK7bY)~;a%FRMY;+<zAR<IXPFY`6K~hIWR9{k6OhqCr3LqdLAR=^Oa%W|9Ut@1>baHQOB03-'
        '<GBz_YG%z?eZf|a1W?^%5UuJJ|cVTj5J0M0$SVcr2NK{l%AW%V6Lr5TIZ*pU0I&^Y%Wgv5JY-MbDAarPHb8cm4VRUJBWn*D;'
        'Wh@|Ncwu97Yi@6MZeea?Wpi|Ia&m8MY+_+!Yan!Ca%W|9J0NavAZKN6WpZg_W^Zz1WoTt=aAk5LED9hXARr=hWo2Yvb98TTB'
        '03;$b!==b3LqdLAR=&bWn*t{WNCD1Z*FrUIv^rLPfblvPG3+`MIb96Wq4s@bZ=s6Wn*+@cxiKVb1!sxaAhxSX>Mg<XJs#IZf'
        '|#Pa%E<7bYXO5AS)nccwu97V_|b;VQg$~cWh~MbUPq!Z((v`X>@X7a(Pu!Omi<`a%XlTED9hXARr=eZ*z2GZ*F91bZKvHb0R'
        't*B12D2O;1i=P)}1-AS)nccwu97b#7y5VQyz-WG`W2b7gLHAa8OYWq4s@bZ%vLQ%FG|a%Ev;Vqs%zAS)nYbaZlQVs&(3Vr*q'
        '}WpZt4ZeeV1b#!BIZDk?~ARv7#3LqeR3LqdLAR=R7b7dkrAR;tkEo5bEWprgCED9hXARr=PWMOc0WpZC-Zgg^aB03-'
        '<aB^>AWnXY|X>Ma_aA9m;WMyn+bY)+2WpZ|9aw04WARr(hB64MBB03-'
        '<a%E<7FKA_9WOFZaWoTt#a&uv9FK}{cZewY1VQepBd0}`lGBqtSFfukVF)%SLa&L8XWg;vJARr(hB4v1RWn*+@WM6M=WFk5s'
        'B4IXUW;bIuIAt<7Wo0-qH#ae3V=y*iFgG|eVKp!~Wn^PvWn?olFd{4pARr(hB6MMLXJvFEIv{R!Y-'
        '}tFARr(hB4v1RWn*+@WM6V+b9HQVB03-<L`6<nUsOR-M@3X$QdLYvA}k6ZARr(jbYXI5WprO-'
        'Z*FvQZ)_quAR;n0GchzUI5uQuY-Mz5Z*C$i3LqdLAR=>RWn^D-bZ>AXIv{R!Y-}tFARr(hB5-'
        'nLV{dL`X>@6CZgV0!AR<FgO-)ZuUr<s-AS)nccwu97Z(?d?V{~PBX>)XQFLZfuWiM=LZe?L-WiM-'
        'PZ+C8TWoC19VRU66D<EZfVPkY-VRL0+Y;13LY-'
        'w|JJ0NavVRB+=baG*Gc~w$Ob1z|XXLcei3LqdLAR=&Yb97^GZe(e6X>V?GB03-'
        '<Lr+ajPflM@Pg7JND<EZfVPkZ4ZewU+Zf9j=FJWSHWo~pJZ*m}Icwu97Ze@2<NI@WSWnpAuVPk6`D<ENXbaH88b#!53Y-'
        'Mz1a&2jDVQg=8bYpLAWg-e7Abl(fARv1RARr(hB4c55Wg<EtA~a(nED9hXARr=PWMOc0WpZC-Zgg^aB03-'
        '<V{Bn_b7^LIUvGDAWpZC~Z*_EKUv6(-'
        'aCLKNA}k6ZARr(ja%E;BIv^r)WoC0PXk}q!b1!mbXk}q?b75>RaB^vGV`*?<Y%gPZVR$hzH7znQGBz<WFflE1Z*_EKA}k6ZA'
        'Rr(jWq5F9V{~O?UvF$=B03-'
        '<VK!uDH)A(AWimHqWjHZ6H!)*lFg9W^H#jq4H8413WMg4vWHT}_A}k6ZARr(jbYXI5WppAsAR;p|VP-fvGC4LmH8o*mHaIjj'
        'F*jv5VKp~nWi@0tVKg%_W;J0sH!&hC3LqdLAR=XWaAjk3Wn^D+Wpj0GbRs$+B2QONMN(f<PgPV!Us6R;RY_7sL|;x%Ur<$3N'
        'Fpo>ARr(hB6MMLXJvF>V{dMBa&K%RIv^r<Xm4y~V{dJ3X>=`eVQyz-'
        'V{Bn_b7^L2WpX<pdS!BNNK{l%ZFO{EbZKvHFM4HiZ%1i#aCLKNA}k6ZARr(jb7f^@UvqSCa3VS&Zgp&IED9hXARr=ea%E$0Z'
        'e(e6X>V?GB03-<Lr+ajPflM@QbiyuAZ2)AV{~s~YGq?|Wq4_GbaO9sd2nSfY-'
        'w&~VP|D8Yi@6MZgOR2b97;JWgsgcWq4s@bYo$2WnpY=Z+C2Ib96f(Zf{|7Vrg`8VRCs@QcQC%VRC17A}k6ZARr(jaBp*TV{d'
        'L`X>@6CZgV0!AR<FgO-)ZuUr<j|R3IxLWq4s@baifHXkl(=Wn?d5Vsm9~bRchXAZ2)AV{~q1cT-'
        '3~AaZ44WMW}sYalBiVRUqIX<~JBVPb4$bY*gFX>MU`Z*_EIZ*65F3LqeTED9hXdkP>RARr=RVRL07Iv^r6WG!NHVQyn+En{+'
        'JVRU68ED9hXARr=PWMOc0WpZC-Zgg^aB03-<V{&C-'
        'bY)*}ZewL%W@&hIb#i4QED9hXARr=gWo9BeAR=;QW^*rSWnpA<FLGsQWnpr2VQepKX=Y_(d1Gv4FK}{iZ)PuJd0}`lGBqtSF'
        'fukVF)%SAED9hXARr=TcyMK7bY)~;Z){{DIv{R!Y-'
        '}tFARr(hB6MMLXJvFEIv^r5GGS&oIWjpmIW;w5WHvZ7H8D43H(@n5V`VjDIAJt1F=jPkIX5vPED9hXARr=TcyMK7bY)~;a%F'
        'RMY;+<zAR<9SLq$+jUr$nBS5Hz)ModpvUqeAqK|)DPNmN;1Q&dk-A}k6ZARr(jbYXI5WprO-'
        'Z*FvQZ)_quAR;n0Gch(eHaKH)Wnpw_Z*DI#HZw6cI5s(HZEbaQVPb4$A}k6ZARr(jb7f^@UvqSCa3VS&Zgp&IED9hXARr=ea'
        '%E$0Ze(e6X>V?GB03-<Lr+ajPflM@QbiyuAZ2)AV{~s~YGq?|Wq4_GbaO9sd2nSfY-'
        'w&~VP|D8Yi@6MZgOR2b97;JWgsgcWq4s@bYo$2WnpY=Z+C2Ib96f(Zf{|7Vrg`8VRCs@QcQC%VRC17A}k6ZARr(jaBp*TV{d'
        'L`X>@6CZgV0!AR<FgO-)ZuUr<j|R3IxLWq4s@baifHXkl(=Wn?d5Vsm9~bRchXAZ2)AV{~q1cT-'
        '3~AaZ44WMW}sYalBiVRUqIX<~JBVPb4$bY*gFX>MU`Z*_EIZ*65F3LqeTED9hXdkP>RARr=RVRL07Iv^r6WG!NHVQyn+Ep>2'
        'YVRU68ED9hXARr=PWMOc0WpZC-Zgg^aB03-'
        '<aB^>AWnXD+ZFO{EVr*q!b#P>1bY&te3LqdLAR=;QW+FNuB64MBb1!IRVPtbJa%E^`VRCa}Y%gqSW@Te}V{BzFaB^>NW-'
        'nuTVR$hzH7znQGBz<WFfk%53LqdLAR=XWaAjk3Wn^D(Y-A!jAR;p|VP-fvGC4LmH8o*mHaIjjF*jv5VKp~nWi@0tVKg%_W;J'
        '0sH!&hC3LqdLAR=^Oa%W|9B03-'
        '<H#9k8GBGh^WnyA8G&E#2H8x~oHDNJkH#B87H#Iq7GBG!1WjSPFA}k6ZARr(jWq5F9V{~O?Uvg!0b!>DZIv^rMMNU~?R6$Zl'
        'MO0r>RZK-'
        '9ED9hXARr=iVRC0>bYEj{Zgg^QY$7@!A~H5JF*Z0hIUsd#WMOn=AaZqVWgv5KWn*b(X=7n*Y<W8%PDVy8Z*FXPAY^53X<=+2'
        'PES-IP(f2uA}k6ZARr(jb7f^@UvqSCa3VS&Zgp&IED9hXARr=ea%E$0Ze(e6X>V?GB03-'
        '<Lr+ajPflM@QbiyuAZ2)AV{~s~YGq?|Wq4_GbaO9sd2nSfY-'
        'w&~VP|D8Yi@6MZgOR2b97;JWgsgcWq4s@bYo$2WnpY=Z+C2Ib96f(Zf{|7Vrg`8VRCs@QcQC%VRC17A}k6ZARr(jaBp*TV{d'
        'L`X>@6CZgV0!AR<FgO-)ZuUr<j|R3IxLWq4s@baifHXkl(=Wn?d5Vsm9~bRchXAZ2)AV{~q1cT-'
        '3~AaZ44WMW}sYalBiVRUqIX<~JBVPb4$bY*gFX>MU`Z*_EIZ*65eAa-GFX=Gz>ZEb0EZ(?d?V{{-'
        '!Mr37kV`Xk+VQzFFZ)Rh1Wnpw>WOQk8EFg1ba&~2Mb#P>1bY*aIZ(?O2W^Zz1WjbbIY;$ESAZ2)PWn*-8aAaY0Wi4`bY-'
        'M(7Z){<7X>V>IZf|r>Mn)nEARv7#3LqeR3LqdLAR=R7b7dkrAR;tmEn;$EZewUIWMyn+bY&te3LqdLAR=L8VQ_S1a$jX`baH'
        'tjIv^r&a&KZ~UukV^b#!53Y-L|$Wo%`1Wg;vJARr(hB64MBB03-<a%E<7FKA_9WOFZaWoTt#a&uv9FKlUMWn+0`Y-'
        'KNSa&K>DFJpOOcrh|HEiy1NHZd_UF(NDqARr(hB4v1RWn*+@WM6M=WFk5sA~P~!W;i)AIW{>pHDP2nI5ageH)S_rH8*2rHDo'
        'wpG&3<~HDNh7F(NDqARr(hB6MMLXJvFEIv{R!Y-}tFARr(hB4v1RWn*+@WM6V+b9HQVB03-<L`6<nUsOR-'
        'M@3X$QdLYvA}k6ZARr(jbYXI5WprO-'
        'Z*FvQZ)_quAR;n0Gch(eHaKH)Wnpw_Z*DI#HZw6cI5s(HZEbaQVPb4$A}k6ZARr(jb7f^@UvqSCa3VS&Zgp&IED9hXARr=ea'
        '%E$0Ze(e6X>V?GB03-<Lr+ajPflM@QbiyuAZ2)AV{~s~YGq?|Wq4_GbaO9sd2nSfY-'
        'w&~VP|D8Yi@6MZgOR2b97;JWgsgcWq4s@bYo$2WnpY=Z+C2Ib96f(Zf{|7Vrg`8VRCs@QcQC%VRC17A}k6ZARr(jaBp*TV{d'
        'L`X>@6CZgV0!AR<FgO-)ZuUr<j|R3IxLWq4s@baifHXkl(=Wn?d5Vsm9~bRchXAZ2)AV{~q1cT-'
        '3~AaZ44WMW}sYalBiVRUqIX<~JBVPb4$bY*gFX>MU`Z*_EIZ*65F3LqeTED9hXdkP>RARr=RVRL07Iv^r6WG!@IXDwrLWnpw'
        '>A}k6ZARr(jVPs)&bY*g1Wo~qGc_KO>B4cu8VRU6*Z*F5{UuJ1|bairNA}k6ZARr(ja%E;BIv^r)WoC0PbYW+6FLGsQWnpr2'
        'VQepHbY*g3bZKvHb1!3gVR$hzH7znQGBz<WFfk%53LqdLAR=XWaAjk3Wn^D(Y-'
        'A!jAZ~SRY%B^OARr(jbYXI5WppAsAR;p|VP-fvGC4LmH8o*mHaIjjF*jv5VKp~nWi@0tVKg%_W;J0sH!&hC3LqdLAR=XWaAj'
        'k3Wn^D+Wpj0GbRs$+B0)n#MNm{<Pf}l3Pf|-'
        'pOix!|LqSkMLP<<XR9RnBR8LSMED9hXARr=iVRC0>bYEj{Zgg^QY$7@!A~H5JF*Z0iFk^CMVRUJ4ZZ9%6Gch(eH!*2#ZFO{E'
        'Vr*q1ED9hXARr=hWo2Yvb98TTB03;$b!==b3LqdLAR=&bWn*t{WNCD1Z*FrUIv^rLPfblvPG3+`MIb96Wq4s@bZ=s6Wn*+@c'
        'xiKVb1!sxaAhxSX>Mg<XJs#IZf|#Pa%E<7bYXO5AS)nccwu97V_|b;VQg$~cWh~MbUPq!Z((v`X>@X7a(Pu!Omi<`a%XlTED'
        '9hXARr=eZ*z2GZ*F91bZKvHb0Rt*B12D2O;1i=P)}1-AS)nccwu97b#7y5VQyz-WG`W2b7gLHAa8OYWq4s@bZ%vLQ%FG|a%E'
        'v;Vqs%zAS)nYbaZlQVs&(3Vr*q}WpZt4ZeeV1b#!BIZDk?~ARv7#3LqeR3LqdLAR=R7b7dkrAR;tmEp%aLEp>2YVRU68ED9h'
        'XARr=PWMOc0WpZC-Zgg^aB03-'
        '<aB^>AWnXD+ZFO{EVr*q!b#P>1bY&te3LqdLAR=;QW+FNuB64MBb1!sZXLB!dWoTt#a&uv9FKKjTa$$67Z*FrhV|ihCF)}qR'
        'GB7eWF)=VPA}k6ZARr(jWq5F9V{~O?UvF$=B03-'
        '<GcsXjI5{#oHaRslVPrNqG&M0dWjA3pH)CZrWH@0oGcjg0VL3N3A}k6ZARr(jbYXI5WppAsAR;$3Ib$+0F=S<8Vly-'
        '}WHmK5WMVa8F=jV3Wi~f8Ibkv}H)dryWMLvK3LqdLAR=XWaAjk3Wn^D+Wpj0GbRs$+B1A<_SzlB^Qb$EpUs6>}MItN;ARr(h'
        'B6MMLXJvF>V{dMBa&K%RIv^r4HZw6cI5#mMb#P>1bY&oNb!=rIb8uy2X=Z6-'
        'VQg%9J0MO*MlEk{Y<VDLWo~I<Y#>fgR3K16Q&S==3LqdLAR=>RWn^D-bZ>AXIv{R!Y-}tFARr(hB5-'
        'nLV{dL`X>@6CZgV0!AR<FgO-)ZuUr<s-AS)nccwu97Z(?d?V{~PBX>)XQFLZfuWiM=LZe?L-WiM-'
        'PZ+C8TWoC19VRU66D<EZfVPkY-VRL0+Y;13LY-'
        'w|JJ0NavVRB+=baG*Gc~w$Ob1z|XXLcei3LqdLAR=&Yb97^GZe(e6X>V?GB03-'
        '<Lr+ajPflM@Pg7JND<EZfVPkZ4ZewU+Zf9j=FJWSHWo~pJZ*m}Icwu97Ze@2<NI@WSWnpAuVPk6`D<ENXbaH88b#!53Y-'
        'Mz1a&2jDVQg=8bYpLAWji2tVQgt+V{dJ3X>@O5YGq?|AVx-HWpiU?Ze(F@bRchLV{&C-'
        'bY*08X>cqcb7gXNWpZ_JWMOn=aB^>AWgup6a${vWW?^h|Wh@|NcyMK7bailKVRU6Ja&>HFc4=>HVRUJ4ZXj-NbWTP_A_^cNe'
        'JlzfAbScRARr(jV_|b;B03-<G-NGwVP`F5Wo%`1Wg;vJARr(hB4K1<aCBvIUuAA|a(N;;AR=&bZ(?O%X>DzFbYWs_WnW}vY-'
        'Mz1A}k6ZARr(ja%E;BIv^r)WoC0PbYW+6FLGsQWnpr2VQepHbY*g3bZKvHb1!3gVR$hzH7znQGBz<WFfk%53LqdLAR=XWaAj'
        'k3Wn^D(Y-A!jAR;p|VP-fvGC4LmH8o*mHaIjjF*jv5VKp~nWi@0tVKg%_W;J0sH!&hC3LqdLAR=^Oa%W|9B03;$b!==b3Lqd'
        'LAR=XWaAjk3Wn^D+Wpj0GbRs$+B1A<_SzlB^Qb$EpUs6>}MItN;ARr(hB6MMLXJvF>V{dMBa&K%RIv^r4HZw6cI5#k3a%Ew3'
        'X>V>XGBz_YHaIsiX>DzFbYWs_Wg;vJARr(hB6DSBWM6Z1Z*U?yAZ~SRY%B^OARr(jaB^j1Z*F91bZKvHb0Rt*B12D2O;1i=P'
        '*O!8D<EZfVPkY}Vrpe$bY*yHb98erba`-PFKlUUWnpJ!FKcdZcW!cJW^;65bY&nbAZ2)AV{~I-'
        'b7f&{Y;SjLX>)WtAZ~AAa$;$8a$$0LRZ>iIFJW?Lb|NeaARr(hB5-'
        'eWbYpLBWNCD1Z*FrUIv^rLPfblvPG3+@Q&b=;AZ2)AV{~<HV`yP+XJuqBVPbP-Zge1Tav)`RVPkY|Wp`6ZK_GHvVPs-qV{0H'
        'QAYpWLa%p09bYWs_Wprh7ZE0>{Y;SdRV{dI`A_^cNeJlzfAbScRARr(jV_|b;B03-<G-'
        'V<z3LqdLAR=L8VQ_S1a$jX`baHtjIv^r&a&KZ~Ut?izZ*FO0VQgP?VP{`sa%Ew3Wg;vJARr(hB64MBB03-'
        '<a%E<7FLYsNb1!mbXk}q?b75>Rb}%k5E-^ASA}k6ZARr(jWq5F9V{~O?UvF$=B03;$b!==b3LqdLAR=^Oa%W|9B03-'
        '<VK!uDH)A(AWimHqWjHZ6H!)*lFg9W^H#jq4H8413WMg4vWHT}_A}k6ZARr(jWq5F9V{~O?Uvg!0b!>DZIv^rMMNU~?R6$Zl'
        'MO0r>RZK-'
        '9ED9hXARr=iVRC0>bYEj{Zgg^QY$7@!A~H5JF*Z0iGGk$GZ*FO0VQgb^Wnpw_Z*CxEcw=mJWMy+faBw0l3LqdLAR=>RWn^D-'
        'bZ>AXIv{R!Y-}tFARr(hB5-nLV{dL`X>@6CZgV0!AR<FgO-)ZuUr<s-AS)nccwu97Z(?d?V{~PBX>)XQFLZfuWiM=LZe?L-'
        'WiM-PZ+C8TWoC19VRU66D<EZfVPkY-VRL0+Y;13LY-'
        'w|JJ0NavVRB+=baG*Gc~w$Ob1z|XXLcei3LqdLAR=&Yb97^GZe(e6X>V?GB03-'
        '<Lr+ajPflM@Pg7JND<EZfVPkZ4ZewU+Zf9j=FJWSHWo~pJZ*m}Icwu97Ze@2<NI@WSWnpAuVPk6`D<ENXbaH88b#!53Y-'
        'Mz1a&2jDVQg=8bYpLAWg-e7Abl(fARv1RARr(hB4c55Wg<EtA~a?#V{&C-'
        'bY&te3LqdLAR=L8VQ_S1a$jX`baHtjIv^r&a&KZ~Uvg!0baH89bY)~;V{&C-'
        'bY&te3LqdLAR=;QW+FNuB64MBb1!IRVPtbJa%E^`VRCa}Y%g+Ub98cPV{~O?FJpOOcrh|HEiy1NHZd_UF)d?qWnpw>A}k6ZA'
        'Rr(jWq5F9V{~O?UvF$=B03;$b!==b3LqdLAR=^Oa%W|9B03-'
        '<VK!uDH)A(AWimHqWjHZ6H!)*lFg9W^H#jq4H8413WMg4vWHT}_A}k6ZARr(jWq5F9V{~O?Uvg!0b!>DZIv^rMMNU~?R6$Zl'
        'MO0r>RZK-9ED9hXARr=iVRC0>bYEj{Zgg^QY$7@!A~H5JF*Z0iG-GmQVRUJ4ZXjiNV{CO~WphDra3U-'
        'UARr(hB6DSBWM6Z1Z*U?yAZ~SRY%B^OARr(jaB^j1Z*F91bZKvHb0Rt*B12D2O;1i=P*O!8D<EZfVPkY}Vrpe$bY*yHb98er'
        'ba`-PFKlUUWnpJ!FKcdZcW!cJW^;65bY&nbAZ2)AV{~I-'
        'b7f&{Y;SjLX>)WtAZ~AAa$;$8a$$0LRZ>iIFJW?Lb|NeaARr(hB5-'
        'eWbYpLBWNCD1Z*FrUIv^rLPfblvPG3+@Q&b=;AZ2)AV{~<HV`yP+XJuqBVPbP-Zge1Tav)`RVPkY|Wp`6ZK_GHvVPs-qV{0H'
        'QAYpWLa%p09bYWs_Wprh7ZE0>{Y;SdRV{dI`A_^cNeJlzfAbScRARr(jV_|b;B03-<G-fSrZ+2xOED9hXARr=PWMOc0WpZC-'
        'Zgg^aB03-<aB^>AWnXe-b98cPV{~O?Uu|!8Wg;vJARr(hB64MBB03-'
        '<a%E<7FKA_9WOFZaWoTt#a&uv9FLGsbbaH89bY)~OV|ihCF)}qRGB7eWF)=VPEp2ahWg;vJARr(hB4v1RWn*+@WM6M=WFk5s'
        'B4#l%W@2JEWn(omWi(?qH8C(XHZ*27Ffw5=HD)q4F*h@2Wj8l6Wg;vJARr(hB6MMLXJvFEIv^rpHe_ZuV>dWuGB;&qI59UjF'
        '=JyeHexV0I5S~2FgRsoV_{`vGcqtDED9hXARr=TcyMK7bY)~;a%FRMY;+<zAR<IXPFY`6K~hIWR9{k6OhqCr3LqdLAR=^Oa%'
        'W|9Ut@1>baHQOB03-<GBz_YHaIslb#P>1bY&oAcw=mJWMy+faBw0l3LqdLAR=>RWn^D-'
        'bZ>AXIv^rVK~6+LR8LY_AWwI0WpZ<6Wn>^^X>)XGZew(1cxiKVX>MmAa%FRLa%p39Wn^t{c4Z)PWo95@bYfw1Wji2KR8LS}P'
        'ghPwQeQzsR7p=xUs6R;RY_7sL?CHqVPbP-Zge{!V_|M?Z*)O$aAsj_Y+_+!Ya%QPARr(hB5-nLV{dL`X>@6CZgV0!AR<FgO-'
        ')ZuUr<s-AS)nccwu97Z(?d?V{~PBX>)XQFLZfuWiM=LZe?L-WiM-PZ+C8TWoC19VRU66D<EZfVPkY-VRL0+Y;13LY-'
        'w|JJ0NavVRB+=baG*Gc~w$Ob1z|XXLcei3LqdLAR=&Yb97^GZe(e6X>V?GB03-'
        '<Lr+ajPflM@Pg7JND<EZfVPkZ4ZewU+Zf9j=FJWSHWo~pJZ*m}Icwu97Ze@2<NI@WSWnpAuVPk6`D<ENXbaH88b#!53Y-'
        'Mz1a&2jDVQg=8bYpLAWg-e7Abl(fARv1RARr(hB4c55Wg<EtB2YCgWp-t5bS-aUb7gXNWg;vJARr(hB4K1<aCBvIUuAA|a(N'
        ';;AR=;QVPs!rcxiKVX>Mm<G+|$1aByE`c4cmKA}k6ZARr(ja%E;BIv^r)WoC0PXk}q!b1!mbXk}q?b75>RaB^vGV`*?<Y%gP'
        'ZVR$hzH7znQGBz<WFflE1Z*_EKA}k6ZARr(jWq5F9V{~O?UvF$=B03-<W-&5mVq!RDV>L2mG-Ee4F)%eYG-ft1GGQ?_W-'
        '>N0H#25sH#ah6A}k6ZARr(jbYXI5WppAsAR=KlWM(&GH#lW7H)UlwF*i3cV`DHjVlX#2GhsC_IAvsGVP#}9GB6@63LqdLAR='
        'XWaAjk3Wn^D+Wpj0GbRs$+B2q;`L|;!%Oj%z;Pf}7(LQhgbR7p=xUrtX{UsO^_M@L0cUq@9zQbb=+K~qyAED9hXARr=iVRC0'
        '>bYEj{Zgg^QY$7@!B1L#%V{{;Ja%FaDZ*_BQc_4CSV{dX~Wn?sAAVF|&AZ2!CZgeklb#6&SJ0NChcx7ZqMN}YVZe(z8X>N3L'
        'Aa8DLc`P7qZy;@TbYXO9Z*DJTcyw}ML2z(zb#rJUED9hXARr=hWo2Yvb98TTB03;$b!==b3LqdLAR=&bWn*t{WNCD1Z*FrUI'
        'v^rLPfblvPG3+`MId2rWFT~9a&2jDVQe&EAaZ46WodA9AaHVJb7gLHJ0NavAZ}%MAZ2!CZge1Ha%Ew3Wn>~O3LqdLAR=&Yb9'
        '7^GZe(e6X>V?GB03-<Qe|OeEpKjYd2eEKWpZ|9av*bcV`F7=b0AMnOj#gMK~qy-'
        'K|@1DP*g=kEFfWQY;<UHWo2b%W@Te^V{dhCbaNnOcwu97Yi@6MZfS0GWoKn_Fd!&yZ**dBZ)_<nAZB4{Y;|&FUvgz(b8l`vZ'
        'gp&IEFg7ma%FRGY<6X2J!WBSb7d@UWq5RBVRL0rH7#UyaBOK~VRU6JW@T<;Wh@|VVRU0?X>Mm<VQ_F?baH8DXJv9<XLVt6WM'
        '5@?WpZO_b7f>bW?^h|WiB92Z*F>Ja&Iqgb!==PZ(?(0a&}>KX>V?GAaHVJb7gXNWn>^>b3#l{LrX<OFH=-'
        'cP%I#IZgORFZ)|pDWIc3pb!9AWWq5RBVRL0YZgp&IJ0NavAaZ4Na(N(cav))3c42N~WiBELARv7#3LqeR3LqdLAR=R7b7dkr'
        'AR<sTEo60YY-wX*bY(4OWo~0-A}k6ZARr(jVPs)&bY*g1Wo~qGc_KO>B64MFWn*+-Wq4_GbZKs9Uvpt?WnXe-adl;LbYEs=Z'
        'ewL4ED9hXARr=gWo9BeAR<IXR7FxvNlr;qR7pc1a%E<7FKA_9WOFZaWoTt#a&uv9FKlUMWn+0`Y-'
        'KNUWpQ<7b98erJaT1mb!BsOUuk4MA}k6ZARr(jWq5F9V{~O?UvF$=B03-'
        '<MOZ;YR3LI?adl;LbY^94V`XD+ZEb0EAaZ46Z*pX1WFTUBAZc!CbZKF1AaZ4Kb!BsOW@T<;Wg;vJARr(hB6MMLXJvFEIv{R!'
        'Y-}tFARr(hB4v1RWn*+@WM6V+b9HQVB03-'
        '<Qbkc!MN?E?Nkm^kOj1QbL|I=|Q$<8yPETJ`MMF<cL|;TnQ&2%vLr5Yl3LqdLAR=^Oa%W|9Ut@1>baHQOB03-'
        '<MR;j*bZKs9AZ2)AV{{;6VQz13X=7n*AaZ4Kb!BsOAZBH5V`U&KAZ2)AV{~(2ZDk;GWpQ<7b95kLd30rSFKA(NXgeTgX?SI1'
        'VPs)&bY*fNWOZ<CX=7n@Wguf{Wn*hBAZ~9UZe@2MW_4_Aa&L8XWgui}b8umFV`w5Q3LqdLAR=>RWn^D-bZ>AXIv{R!Y-'
        '}tFARr(hB5-nLV{dL`X>@6CZgV0!AR<FgO-)ZuUr<s-'
        'J0Ngxb7^#Gc4cgNAaZ44WFTdDVPkY_Zf|#PAZ2)Ib98BLXCP)}ZewLHZ(?d?V{|WaWpQ<7b96f(b75^|AaZ4Kb!BsONklJjY'
        '+-'
        'I7WNBw*b96f(Zf_uUZfkCDcWxkWa%pF2ZeeV9a%psBA}k6ZARr(jaBp*TV{dL`X>@6CZgV0!AR<$BV`F7=b5Bl8Svnw4K~qy'
        '-L`6<XMMPgzK~hIWR9{0+PE=A)Oe`Q}cwu97AaHMUX>@6JWo&sMZDDj{Xk}y|Wq4_GbZKs9W@T<;WiNAKZDnI&Zf|aBV_|G^'
        'WpQ<7b96~WD{yRKZY&^VX>)L4bYo~QK~PC|a%psBFGp!~aCLKNAY*TJZgg`XWq4s@bZKsMWoKn_Fd!&yZ**p1Y;$EPEFg7ma'
        '%FRGY<6X2J!WBSb7d@UWq5RBVRL0PXD%Q{VQFl2a%C@dZfkCDcWyc$aB^jHWpZ|9AYo&4bzy97Z*Fd7XJK?{c4cF4b#8QXAa'
        '8PRb!=>PZfkCDcWx{oW@T<;WnXi2VRU66Rz*@tMoC3PUrj+&Lr5<{LQ_RfR4-'
        'LdOHNN$PA^SKQ%ylsLr5TDZe&wPL2q(yb!==bAaZ44b8l`eAa!nXWpi(Ac4cHebaHiNEN*3ZbYo$2Wj$_nY-'
        '~FqLQGFXOGQL1AZ~9Ua%FUKc`svcZE$67b76F8Z*DJPWOiY0V`VNP3LqeTED9hXdkP>RARr=RVRL07Iv^r6XCf>LARr(hB4K'
        '1<aCBvIUuAA|a(N;;AR=;QbairWUvGDAWpZC(V{~b6ZXzrSARr(hB64MBB03-'
        '<a%E<7FKA_9WOFZaWoTt#a&uv9FK}{cZewY1VQepBd0}`lGBqtSFfukVF)%SLa&L8XWg;vJARr(hB4v1RWn*+@WM6M=WFk5s'
        'B4IXUW;bIuIAt<7Wo0-'
        'qH#ae3V=y*iFgG|eVKp!~Wn^PvWn?olFd{4pARr(hB6MMLXJvFEIv^r5GGS&oIWjpmIW;w5WHvZ7H8D43H(@n5V`VjDIAJt1'
        'F=jPkIX5vPED9hXARr=TcyMK7bY)~;a%FRMY;+<zAR<##Pf%Y^S58GzUqM4uNl#8+Qbkc!Nm4~bA}k6ZARr(jbYXI5WprO-'
        'Z*FvQZ)_quAR<q9Ze?<1cwu97Mn-gKWo}MJMsj6pWn*+@WHL50F*GnZHaj44WnpAuVPk7gcWz~JVPkY}av*YLadl~OWn>~O3'
        'LqdLAR=>RWn^D-bZ>AXIv{R!Y-}tFARr(hB5-nLV{dL`X>@6CZgV0!AR<FgO-)ZuUr<s-'
        'AS)nccwu97Z(?d?V{~PBX>)XQFLZfuWiM=LZe?L-WiM-PZ+C8TWoC19VRU66D<EZfVPkY-VRL0+Y;13LY-'
        'w|JJ0NavVRB+=baG*Gc~w$Ob1z|XXLcei3LqdLAR=&Yb97^GZe(e6X>V?GB03-'
        '<Lr+ajPflM@Pg7JND<EZfVPkZ4ZewU+Zf9j=FJWSHWo~pJZ*m}Icwu97Ze@2<NI@WSWnpAuVPk6`D<ENXbaH88b#!53Y-'
        'Mz1a&2jDVQg=8bYpLAWg-e7Abkp53I'
    ),
    'status_aware_vectors': (
        'TM8f`dkP>RARr=cVQpn1Iv^rvZ*OERWOZ<CX=7n@Wg;vJARr(hB64MPb#iVZIv{%rARr(hARr=hV`yb<VPAG-'
        'a&u{KZX!A$F)RuoARr(hAR=;Qadl;LbYE#?B03-<V{dMAbaHiLbZKvHEpRn0Ffk%53LqdLARr(jaBN|2Uvp?-'
        'GBq|LIv^r3F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YBED9hXA'
        'Rr(hB4c55WnXDzB03-<P&F-'
        'Nb#QEHV_|e<EoNnIV`U;N3LqdLARr(jb97;Jb#o#*AR<sfQ&V3=MNUaYL|;@vQb$EpUqeq$R8mh&A}k6ZARr(hAR=XUX=G(?'
        'V`X1+XkjunHX=G8A~G^EGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUG'
        '9oMrARr(hARr=cWq5R7V_|b;B03-<G-'
        'o0#3LqdLARr(jb#8KHb8l>RWn>~cAZB4~b7d?FARr(hARr=eHD7aeVsd43b!>DZIv{%rARr(hARr(hB5h%GV`yn^XJ27(a9?'
        'zCX=i6;a$je4VRB?&Wq4(BV`+0`WFk5sW?^h|Wh@FHARr(hARr(jZ(?(0a&~28UvFk+Uvgz;UuAY>Zge6#AZ~SRY%B^OARr('
        'hARr(jb7f<1Ze(9%a%pyDa$jU=b8umFV`yJvZ*^{TB03;2ED9hXARr(hARr=PaA{v{b#!5LX>V>{V{dhCbRs$+Ff0lnARr(h'
        'ARr(jXK8d_aCLKNUt@1|Zge6#ATTTnARr(hARr(hB4v1Kb98BLXJ2MzZewL%b7)~AIv^r5Gcz+YGcz+YGcz+YGcz+YGcz+YG'
        'cz+YGcz+YGcz+YGcz+YGcz+HED9hXARr(hARr=UWo~0-'
        'UvqR}bY&tsAR<;pQb|ThMMPgsK~zIXA}k6ZARr(hARr(ja%FLKWpi|2aBN|2Uu|J@V`w5eAartdWh@FHARr(hARr(jW?^Y;b'
        '#i52a%Ew2Z*C$wAZ~SRYziPCARr(heF`8TARv7#3LqdLAR=XWaAjk3Wn^D=VQgt+B03;+a&=`a3LqdLAR=L7bai2DUv^<^X='
        'EZgAartdWeOl5eJlzfAbScRARr(jZeeX@B03-'
        '<b#7&NaAjk3Wn?X6X>)L4bYo~DED9hXARr=gWps6NZX!A$dkP>RARr(hB6DMCWo=<!c4cyNX>V>KIv_DD3LqdLARr(ja%FLK'
        'Wpi|2X=EZgAR=RLZgX^Ubz^jCZ*DDcH7zhPA}k6ZARr(hAR=&VVQyb@XkjunHX=G8A~7*BF)=YQF)=YQF)=YQF)=YQF)=YQF'
        ')=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF(NDqARr(hARr=RVRL0)X=EZgAR<sTEo60YY-wX*bY(4OWo'
        '~0-A}k6ZARr(hAR=>gVRUtKB03-'
        '<LQGFXOGQLqRZc}%P(?#jMMPggLqkPSR6$NdMItN;ARr(hARr=Tc4=f~ZewL%b7)~QH8vtTAR;m{GBPqUGBPqUGBPqUGBPqU'
        'GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBP483LqdLARr(jZe@6MUt?i&Wg<EtZgp&IED9hXA'
        'Rr(hB6V(ZWpi(Ac4cHDIv{j%b!99HARr(hARr=eHD7aeVsd43b!>DZIv{%rARr(hARr(hB5h%GV`yn^XJ27(a9?zCX=i6;a$'
        'je4VRB?&Wq4(BV`+0`WFk5sW?^h|Wh@FHARr(hARr(jZ(?(0a&~28UvFk+Uvgz;UuAY>Zge6#AZ~SRY%B^OARr(hARr(jb7f'
        '<1Ze(9%a%pyDa$jU=b8umFV`yJvZ*^{TB03;3ED9hXARr(hARr=PaA{v{b#!5LX>V>{V{dhCbRs$+Ff0lnARr(hARr(jXK8d'
        '_aCLKNUt@1|Zge6#ATTTnARr(hARr(hB4v1Kb98BLXJ2MzZewL%b7)~AIv^r5Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+Y'
        'Gcz+YGcz+YGcz+HED9hXARr(hARr=UWo~0-'
        'UvqR}bY&tsAR<;pQb|ThMMPgsK~zIXA}k6ZARr(hARr(ja%FLKWpi|2aBN|2Uu|J@V`w5eAartdWh@FHARr(hARr(jW?^Y;b'
        '#i52a%Ew2Z*C$wAR={cWq5F9V{~O?Eo5nPaA9;~Xd((AARr(hAbkoTARr)pED9hXARr=TcyMK7bY)~;c42I3WFk5sbaHiNED'
        '9hXARr=PV{~<4Y+rU^Y-wa7Iv{j%b!7@5Abl(fARv1RARr(hB5q-'
        'AWg<EtB6V(McyMK7bY)~MWNCA7VRU0?En{JBZf|rgaA9+EA}k6ZARr(ja%FUNa&96zAbScRARr(hAR=>PXk~3-'
        'Uv_13b7^mGB03;3ED9hXARr(hB64MMb!BsOUuk3_Iv^rrZ*FsRa&=>LX>V>Va5XJ3F(NDqARr(hARr=eY+-'
        'I+b7)~QH8vtTAR;j_F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)<='
        '63LqdLARr(jV_|b;Uuk3_Iv^rYH7#UyaBOK~VRU6JW@T<;Wg;vJARr(hARr=hbYXOLb0Rt*B2Yn7Q(r_yPDw>XUsOR-'
        'M@3X$Lr+drQcp}GED9hXARr(hB4u`IWMyt+WnXh>VKOx~B03-'
        '<GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUA}k6ZARr(hAR=yM'
        'cywQ5VRL07Iv^r6XCf>LARr(hARr=jZgORFZ)|pDWFk5sW?^h|Wh@FHARr(hAR=%zUvqV0a%FRMY;+<zAbScRARr(hARr(jZ'
        'DDj{XlZU|Utw@?UvzS5XJ=({UuSh;a%5j+cx7^9X>(;{B03;uVQh0{ED9hXARr(hARr=dVsmA3c4cH=Z)Rp+a%E;;Wp-'
        't5bRs$+Zgp&IED9hXARr(hARr=hWn*t{WM5=*X?A6DUu0=>aA9;~XkTM*b#8PbIv_DD3LqdLARr(hAR=LKX<u!1bYXO9Z*E^'
        '>Z*^{TB03;2ED9hXARr(hARr=VX>?z3b#rK6V{dhCbRs$+Ff0lnARr(hARr(jWq4_GbZKs9UuI=)V`X1+Xkj8cAR;p}Gcz+Y'
        'Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGczJA3LqdLARr(hAR=aEZewL%b97;JWg<EtB34CGNk&OUL|;uoR6|H2E'
        'D9hXARr(hARr=gWpQ<7b97&DY+-'
        'I+ZDDj{Xd*fwbaHiNED9hXARr(hARr=UVQFl2a%Ep~Wnpt~ZX!A$B6V(McyMK7bY)~MWNCA7VRU0?A_^cNARr(heF`8TARv7'
        '#3LqdLAR=XWaAjk3Wn^D=VQgt+B03;uVQh0{ED9hXARr=PV{~<4Y+rU^Y-'
        'wa7Iv{3YY;$D_ARv7#3LqeR3LqdLAR=yIZDk@lAR={cWq5F9V{~O?Eq8KhbY&te3LqdLAR=;QbairWB03;@3LqdLARr(jb7N'
        '>_ZDC(_WpZ<AZ*C$wATcZoARr(hARr=gWpQ<7b97&6WFk5sB4clEb98cbV{~b6ZY^*%Eif@6ED9hXARr(hB5-'
        'VBZeMd~VKOx~B03-'
        '<F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQA}k6ZARr(hAR=R7'
        'b7fy?WFk5sB2YCgWOZ<CX=7n@Wi4i9ZewL4ED9hXARr(hB6D<MbaitgIv^rKOix2gMMPg!PDNNyMMG3YL|;KeLq$+jK~6(OA'
        '}k6ZARr(hAR=XUX=G(?V`X1+XkjunHX=G8A~G^EGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGB'
        'PqUGBPqUGBPqUGBPqUG9oMrARr(hARr=cWq5R7V_|b;B03;$b!==b3LqdLARr(jb#8KHb8l>RWn>~cAartdWh@FHARr(hAR='
        '%zUvqV0a%FRMY;+<zAbScRARr(hARr(jZDDj{XlZU|Utw@?UvzS5XJ=({UuSh;a%5j+cx7^9X>(;{B03;uVQh0{ED9hXARr('
        'hARr=dVsmA3c4cH=Z)Rp+a%E;;Wp-'
        't5bRs$+Zgp&IED9hXARr(hARr=hWn*t{WM5=*X?A6DUu0=>aA9;~XkTM*b#8PbIv_AC3LqdLARr(hAR=LKX<u!1bYXO9Z*E^'
        '>Z*^{TB03;3ED9hXARr(hARr=VX>?z3b#rK6V{dhCbRs$+Ff0lnARr(hARr(jWq4_GbZKs9UuI=)V`X1+Xkj8cAR;p}Gcz+Y'
        'Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGczJA3LqdLARr(hAR=aEZewL%b97;JWg<EtB34CGNk&OUL|;uoR6|H2E'
        'D9hXARr(hARr=gWpQ<7b97&DY+-'
        'I+ZDDj{Xd*fwbaHiNED9hXARr(hARr=UVQFl2a%Ep~Wnpt~ZX!A$B6V(McyMK7bY)~McXDZTWg-'
        'e7ARr(hAbkoTARr)pED9hXARr=TcyMK7bY)~;c42I3WFk5sbaHiNED9hXARr=PV{~<4Y+rU^Y-'
        'wa7Iv{j%b!7@5Abl(fARv1RARr(hB5q-'
        'AWg<EtB6V(McyMK7bY)~McXDZTWi4Z2Zf<XMEpTCTb0RDXARr(hB64MPb#iVZIv{%rARr(hARr=hV`yb<VPAG-'
        'a&u{KZX!A$F)RuoARr(hAR=;Qadl;LbYE#?B03-<V{dMAbaHiLbZKvHEpRn0Ffk%53LqdLARr(jaBN|2Uvp?-'
        'GBq|LIv^r3F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YBED9hXA'
        'Rr(hB4c55WnXDzB03-<P&F-'
        'Nb#QEHV_|e<EoNnIV`U;N3LqdLARr(jb97;Jb#o#*AR<sfQ&V3=MNUaYL|;@vQb$EpUqeq$R8mh&A}k6ZARr(hAR=XUX=G(?'
        'V`X1+XkjunHX=G8A~G^EGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUG'
        '9oMrARr(hARr=cWq5R7V_|b;B03-<G-'
        'o0#3LqdLARr(jb#8KHb8l>RWn>~cAZB4~b7d?FARr(hARr=eHD7aeVsd43b!>DZIv{%rARr(hARr(hB5h%GV`yn^XJ27(a9?'
        'zCX=i6;a$je4VRB?&Wq4(BV`+0`WFk5sW?^h|Wh@FHARr(hARr(jZ(?(0a&~28UvFk+Uvgz;UuAY>Zge6#AZ~SRY%B^OARr('
        'hARr(jb7f<1Ze(9%a%pyDa$jU=b8umFV`yJvZ*^{TB03;2ED9hXARr(hARr=PaA{v{b#!5LX>V>{V{dhCbRs$+F)RuoARr(h'
        'ARr(jXK8d_aCLKNUt@1|Zge6#ATTTnARr(hARr(hB4v1Kb98BLXJ2MzZewL%b7)~AIv^r5Gcz+YGcz+YGcz+YGcz+YGcz+YG'
        'cz+YGcz+YGcz+YGcz+YGcz+HED9hXARr(hARr=UWo~0-'
        'UvqR}bY&tsAR<;pQb|ThMMPgsK~zIXA}k6ZARr(hARr(ja%FLKWpi|2aBN|2Uu|J@V`w5eAartdWh@FHARr(hARr(jW?^Y;b'
        '#i52a%Ew2Z*C$wAR={cWq5F9V{~O?Eq8KhbY&t6ARr(hARv7TARr(heJlzfARr(jWq5F9V{~O?Uv^<^X=EZgAZB4~b7d?FAR'
        'r(hB4J~6bzy8@c42I3WFk5sW?^h|WeOl5eJlzfAbScRARr(jZeeX@B03-'
        '<b#7~JZ+C7jV{dhCbRsMYARr(hB64MPb#iVZIv{%rARr(hARr=hV`yb<VPAG-'
        'a&u{KZX!A$F)RuoARr(hAR=;Qadl;LbYE#?B03-<V{dMAbaHiLbZKvHEpRn0Ffk%53LqdLARr(jaBN|2Uvp?-'
        'GBq|LIv^r3F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YBED9hXA'
        'Rr(hB4c55WnXDzB03-<P&F-Nb#QEHV_|e<EoNnIV`U;N3LqdLARr(jb97;Jb#o#*AR<CcPeV&ZL|;`-'
        'OHNN$P9iJ{ARr(hARr=Tc4=f~ZewL%b7)~QH8vtTAR;m{GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUG'
        'BPqUGBPqUGBPqUGBPqUGBPqUGBP483LqdLARr(jZe@6MUt?i&Wg<EtZgp&IED9hXARr(hB6V(ZWpi(Ac4cHDIv{j%b!99HAR'
        'r(hARr=eHD7aeVsd43b!>DZIv{%rARr(hARr(hB5h%GV`yn^XJ27(a9?zCX=i6;a$je4VRB?&Wq4(BV`+0`WFk5sW?^h|Wh@'
        'FHARr(hARr(jZ(?(0a&~28UvFk+Uvgz;UuAY>Zge6#AZ~SRY%B^OARr(hARr(jb7f<1Ze(9%a%pyDa$jU=b8umFV`yJvZ*^{'
        'TB03;2ED9hXARr(hARr=PaA{v{b#!5LX>V>{V{dhCbRs$+Zgp&IED9hXARr(hARr=VX>?z3b#rK6V{dhCbRs$+Ff0lnARr(h'
        'ARr(jWq4_GbZKs9UuI=)V`X1+Xkj8cAR;p}Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGczJA3LqdLARr(hA'
        'R=aEZewL%b97;JWg<EtB34CGNk&OUL|;uoR6|H2ED9hXARr(hARr=gWpQ<7b97&DY+-'
        'I+ZDDj{Xd*fwbaHiNED9hXARr(hARr=UVQFl2a%Ep~Wnpt~ZX!A$B6V(SZf|#PEn{zWZge6FARr(hARv7TARr(heJlzfARr('
        'jWq5F9V{~O?Uv^<^X=EZgAartdWh@FHARr(jVPkZ4VQgP^VQgt+B03;+a&=`2ARv7#3LqeR3LqdLAR=yIZDk@lAR={cYi@6M'
        'ZY^VPb#8PmV_|M?Z*(niVRLgLED9hXARr=gWps6NZX!A$dkP>RARr(hB6DMCWo=<!c4cyNX>V>KIv_DD3LqdLARr(ja%FLKW'
        'pi|2X=EZgAR=RLZgX^Ubz^jCZ*DDcH7zhPA}k6ZARr(hAR=&VVQyb@XkjunHX=G8A~7*BF)=YQF)=YQF)=YQF)=YQF)=YQF)'
        '=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF(NDqARr(hARr=RVRL0)X=EZgAR<sTEo60YY-wX*bY(4OWo~'
        '0-A}k6ZARr(hAR=>gVRUtKB03-<P(f2uUqnStNkv3oR6$ZlMO0ryPfk=)PfQ{#3LqdLARr(jWp-&~Wo~0-Uvp?-'
        'GBq|LIv^r4GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqEED9hXA'
        'Rr(hB5q}PbYEj(b7dkrAR;tpA}k6ZARr(hAR={ca%FRGY<6X2B03;uVQh0{ED9hXARr(hB5*Zdb9G{JWpj0GbRs$+dkP>RAR'
        'r(hARr=bVRU0?X>Mm<VQ_F?baH8DXJv9<XLVt6WM5@?WpZO_b7f>AIv{3YY;$ES3LqdLARr(hAR=#Kb7gXNWn^D(W@cY<WoB'
        'Pxc4cmKB03;$b!==b3LqdLARr(hAR=>RV{dL`Uu1G=c4cy3WNCA7VRU0?Ut@1|Zge6#ATTTnARr(hARr(hB4Kc8Uu|`CVRUJ'
        '4ZeL?>b#8PbIv{R!Y-}tFARr(hARr(hB4=rIUvPDEXkTM*b#8PbIv_AC3LqdLARr(hAR=XWX>)XGZf9R+Wo~0-Uvp?-B03-'
        '<Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YA}k6ZARr(hARr(jW@T<;WnXi2VRU68Iv^reMN&ydNkv3o'
        'O+i#cNFpo>ARr(hARr(hB64MMb!BsOUvO+;ZeML-bYo~DIv{j%b!99HARr(hARr(hB4%M}Y;|&FUvgz(b8l`UIv^r-'
        'ZfkCDcWy0XZ*^{TA_^cNARr(heF`8TARv7#3LqdLAR=XWaAjk3Wn^D=VQgt+B03;uVQh0{ED9hXARr=PV{~<4Y+rU^Y-'
        'wa7Iv{3YY;$D_ARv7#3LqeR3LqdLAR=yIZDk@lAR=aEZewLFK|)hSPE;Z+3LqdLAR=;QbairWB03;@3LqdLARr(jb7N>_ZDC'
        '(_WpZ<AZ*C$wATcZoARr(hARr=gWpQ<7b97&6WFk5sB4clEb98cbV{~b6ZY^*%Eif@6ED9hXARr(hB5-VBZeMd~VKOx~B03-'
        '<F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQA}k6ZARr(hAR=R7'
        'b7fy?WFk5sB2YCgWOZ<CX=7n@Wi4i9ZewL4ED9hXARr(hB6D<MbaitgIv^rKOix2gMMPgtPgGx4MN&ydNkKwPMItN;ARr(hA'
        'Rr=Tc4=f~ZewL%b7)~QH8vtTAR;m{GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGB'
        'PqUGBPqUGBP483LqdLARr(jZe@6MUt?i&Wg<EtZgp&IED9hXARr(hB6V(ZWpi(Ac4cHDIv{j%b!99HARr(hARr=eHD7aeVsd'
        '43b!>DZIv{%rARr(hARr(hB5h%GV`yn^XJ27(a9?zCX=i6;a$je4VRB?&Wq4(BV`+0`WFk5sW?^h|Wh@FHARr(hARr(jZ(?('
        '0a&~28UvFk+Uvgz;UuAY>Zge6#AZ~SRY%B^OARr(hARr(jb7f<1Ze(9%a%pyDa$jU=b8umFV`yJvZ*^{TB03;2ED9hXARr(h'
        'ARr=PaA{v{b#!5LX>V>{V{dhCbRs$+Ff0lnARr(hARr(jXK8d_aCLKNUt@1|Zge6#ATTTnARr(hARr(hB4v1Kb98BLXJ2MzZ'
        'ewL%b7)~AIv{R!Y-}tFARr(hARr(hB4%Z7V`X1+bYXO5B03-<K|)hSPE;Z+3LqdLARr(hAR=;Qadl;LbYF06VQyb-'
        'VRU0?B03;$b!==b3LqdLARr(hAR=aAX>4_JWnXe-VRLV8B03-'
        '<W@T<;Wi3HMQ$<cxA_^cNARr(heF`8TARv7#3LqdLAR=XWaAjk3Wn^D=VQgt+B03;+a&=`a3LqdLAR=L7bai2DUv^<^X=EZg'
        'AartdWeOl5eJlzfAbScRARr(jZeeX@B03-<W@T<;Wi3@sOHNN$P9iJ{ARr(hB64MPb#iVZIv{%rARr(hARr=hV`yb<VPAG-'
        'a&u{KZX!A$F)RuoARr(hAR=;Qadl;LbYE#?B03-<V{dMAbaHiLbZKvHEpRn0Ffk%53LqdLARr(jaBN|2Uvp?-'
        'GBq|LIv^r3F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YBED9hXA'
        'Rr(hB4c55WnXDzB03-<P&F-'
        'Nb#QEHV_|e<EoNnIV`U;N3LqdLARr(jb97;Jb#o#*AR<CcPeV&ZL|;x%R9{v_Qb|ThK|)MLA}k6ZARr(hAR=XUX=G(?V`X1+'
        'XkjunHX=G8A~G^EGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUG9oMrA'
        'Rr(hARr=cWq5R7V_|b;B03;$b!==b3LqdLARr(jb#8KHb8l>RWn>~cAartdWh@FHARr(hAR=%zUvqV0a%FRMY;+<zAbScRAR'
        'r(hARr(jZDDj{XlZU|Utw@?UvzS5XJ=({UuSh;a%5j+cx7^9X>(;{B03;uVQh0{ED9hXARr(hARr=dVsmA3c4cH=Z)Rp+a%E'
        ';;Wp-'
        't5bRs$+Zgp&IED9hXARr(hARr=hWn*t{WM5=*X?A6DUu0=>aA9;~XkTM*b#8PbIv_AC3LqdLARr(hAR=LKX<u!1bYXO9Z*E^'
        '>Z*^{TB03;2ED9hXARr(hARr=VX>?z3b#rK6V{dhCbRs$+Ff0lnARr(hARr(jWq4_GbZKs9UuI=)V`X1+Xkj8cAZ~SRY%B^O'
        'ARr(hARr(jW@T<;WnXi2VRU68Iv^rdPD@TtS56`<3LqdLARr(hAR=;Qadl;LbYF06VQyb-'
        'VRU0?B03;$b!==b3LqdLARr(hAR=aAX>4_JWnXe-VRLV8B03-'
        '<W@T<;Wi3@sOHNN$P9h2*ARr(hAbkoTARr)pED9hXARr=TcyMK7bY)~;c42I3WFk5sbaHiNED9hXARr=PV{~<4Y+rU^Y-'
        'wa7Iv{j%b!7@5Abl(fARv1RARr(hB5q-AWg<EtB5rSPP&F-'
        'Xb!==RED9hXARr=gWps6NZX!A$dkP>RARr(hB6DMCWo=<!c4cyNX>V>KIv_DD3LqdLARr(ja%FLKWpi|2X=EZgAR=RLZgX^U'
        'bz^jCZ*DDcH7zhPA}k6ZARr(hAR=&VVQyb@XkjunHX=G8A~7*BF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF'
        ')=YQF)=YQF)=YQF)=YQF)=YQF)=YQF(NDqARr(hARr=RVRL0)X=EZgAR;tjA}k6ZARr(hAR=>gVRUtKB03-'
        '<P(f2uUqM4dMNm{lL?SE-ARr(hARr=Tc4=f~ZewL%b7)~QH8vtTAR;m{GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqU'
        'GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBP483LqdLARr(jZe@6MUt?i&Wg<EtA~a$xZf0g8ED9hXARr(hB6V(ZWpi(Ac'
        '4cHDIv{3YY;$ES3LqdLARr(ja5Z0Zbz*X5b9HQVB03;$b!==3ARr(heJlzfARr(jWq5F9V{~O?Uv^<^X=EZgAartdWh@FHAR'
        'r(jVPkZ4VQgP^VQgt+B03;+a&=`2ARv7#3LqeR3LqdLAR=yIZDk@lAR=aOa$;#@WMytGY-'
        'x67EoXIMa%3Vb3LqdLAR=;QbairWB03;@3LqdLARr(jb7N>_ZDC(_WpZ<AZ*C$wATcZoARr(hARr=gWpQ<7b97&6WFk5sB4c'
        'lEb98cbV{~b6ZY^*%Eif@6ED9hXARr(hB5-VBZeMd~VKOx~B03-'
        '<F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQA}k6ZARr(hAR=R7'
        'b7fy?WFk5sB2YCgWOZ<CX=7n@Wi4i9ZewL4ED9hXARr(hB6D<MbaitgIv^rYK~qy-'
        'L`6<XMMPgzK~hIWR9{0+PE=A)Od>1_ARr(hARr=Tc4=f~ZewL%b7)~QH8vtTAR;m{GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUG'
        'BPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBP483LqdLARr(jZe@6MUt?i&Wg<EtA~a_rED9hXARr(hB6V(ZWp'
        'i(Ac4cHDIv{3YY;$ES3LqdLARr(ja5Z0Zbz*X5b9HQVB03;@3LqdLARr(hAR=vHbYo~~Zf9R%aByFAa%pF0WpZC<bzyR3UuA'
        'e@a${+8Wn>~cAartdWh@FHARr(hARr(jZ(?(0a&~28UvFk+Uvgz;UuAY>Zge6#AZ~SRY%B^OARr(hARr(jb7f<1Ze(9%a%py'
        'Da$jU=b8umFV`yJvZ*^{TB03;2ED9hXARr(hARr=PaA{v{b#!5LX>V>{V{dhCbRs$+Ff0lnARr(hARr(jXK8d_aCLKNUt@1|'
        'Zge6#ATTTnARr(hARr(hB4v1Kb98BLXJ2MzZewL%b7)~AIv^r5Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YG'
        'cz+HED9hXARr(hARr=UWo~0-'
        'UvqR}bY&tsAR<;pQb|ThMMPgsK~zIXA}k6ZARr(hARr(ja%FLKWpi|2aBN|2Uu|J@V`w5eAartdWh@FHARr(hARr(jW?^Y;b'
        '#i52a%Ew2Z*C$wAZ~SRYziPCARr(heF`8TARv7#3LqdLAR=XWaAjk3Wn^D=VQgt+B03;uVQh0{ED9hXARr=PV{~<4Y+rU^Y-'
        'wa7Iv{3YY;$D_ARv7#3LqeR3LqdLAR=yIZDk@lAR=OKZ)`1YZ*(nVZ*^{TA}k6ZARr(ja%FUNa&96zAbScRARr(hAR=>PXk~'
        '3-Uv_13b7^mGB03;3ED9hXARr(hB64MMb!BsOUuk3_Iv^rrZ*FsRa&=>LX>V>Va5XJ3F(NDqARr(hARr=eY+-'
        'I+b7)~QH8vtTAR;j_F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)<='
        '63LqdLARr(jV_|b;Uuk3_Iv^rYH7#UyaBOK~VRU6JW@T<;Wg;vJARr(hARr=hbYXOLb0Rt*B2Yn7Q(r_yPDw>XUsOR-'
        'M@3X$Lr+drQcp}GED9hXARr(hB4u`IWMyt+WnXh>VKOx~B03-'
        '<GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUA}k6ZARr(hAR=yM'
        'cywQ5VRL07Iv^r6XCf>LARr(hARr=jZgORFZ)|pDWFk5sW?^h|Wh@FHARr(hAR=%zUvqV0a%FRMY;+<zAbScRARr(hARr(jZ'
        'DDj{XlZU|Utw@?UvzS5XJ=({UuSh;a%5j+cx7^9X>(;{B03;uVQh0{ED9hXARr(hARr=dVsmA3c4cH=Z)Rp+a%E;;Wp-'
        't5bRs$+Zgp&IED9hXARr(hARr=hWn*t{WM5=*X?A6DUu0=>aA9;~XkTM*b#8PbIv_AC3LqdLARr(hAR=LKX<u!1bYXO9Z*E^'
        '>Z*^{TB03;uVQh0{ED9hXARr(hARr=VX>?z3b#rK6V{dhCbRs$+Ff0lnARr(hARr(jWq4_GbZKs9UuI=)V`X1+Xkj8cAR;p}'
        'Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGczJA3LqdLARr(hAR=aEZewL%b97;JWg<EtB34CGNk&OUL|;uoR'
        '6|H2ED9hXARr(hARr=gWpQ<7b97&DY+-'
        'I+ZDDj{Xd*fwbaHiNED9hXARr(hARr=UVQFl2a%Ep~Wnpt~ZX!A$Zgp&I3LqdLARr)p3LqdLAbl(fARr(hB4v1RWn*+@WM6h'
        '+Y-wa7Iv{3YY;$ES3LqdLAR=L7bai2DUv^<^X=EZgAZB4~b7cx3Abl(fARv1RARr(hB5q-'
        'AWg<EtB4S}=En#G8VPj=(V|gMh3LqdLAR=;QbairWB03;@3LqdLARr(jb7N>_ZDC(_WpZ<AZ*C$wATcZoARr(hARr=gWpQ<7'
        'b97&6WFk5sB4clEb98cbV{~b6ZY^*%Eif@6ED9hXARr(hB5-VBZeMd~VKOx~B03-'
        '<F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQA}k6ZARr(hAR=R7'
        'b7fy?WFk5sB2YCgWOZ<CX=7n@Wi4i9ZewL4ED9hXARr(hB6D<MbaitgIv^rYK~qy-'
        'L`6<XMMPgzK~hIWR9{0+PE=A)Od>1_ARr(hARr=Tc4=f~ZewL%b7)~QH8vtTAR;m{GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUG'
        'BPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBP483LqdLARr(jZe@6MUt?i&Wg<EtA~azlED9hXARr(hB6V(ZWp'
        'i(Ac4cHDIv{3YY;$ES3LqdLARr(ja5Z0Zbz*X5b9HQVB03;@3LqdLARr(hAR=vHbYo~~Zf9R%aByFAa%pF0WpZC<bzyR3UuA'
        'e@a${+8Wn>~cAZB4~b7d?FARr(hARr(hB5z`IWpZ|9WM6M)W?yn;W?yA?Wo~pLIv{R!Y-'
        '}tFARr(hARr(hB6DS9Z*F8?WO8YCWpZC+X>)L4bYo~=V{dhCbRs$+Ff0lnARr(hARr(jVQ^_*ZFO{EbZKvHUt@1|Zge6#ATT'
        'TnARr(hARr(hB4=rIUvPDEXkTM*b#8PbIv_AC3LqdLARr(hAR=XWX>)XGZf9R+Wo~0-Uvp?-B03-'
        '<Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YA}k6ZARr(hARr(jW@T<;WnXi2VRU68Iv^reMN&ydNkv3o'
        'O+i#cNFpo>ARr(hARr(hB64MMb!BsOUvO+;ZeML-bYo~DIv{j%b!99HARr(hARr(hB4%M}Y;|&FUvgz(b8l`UIv{R!Y-'
        '|c3ARr(hAbkoTARr)pED9hXARr=TcyMK7bY)~;c42I3WFk5sW?^h|Wh@FHARr(jVPkZ4VQgP^VQgt+B03;uVQh0{3LqeTED9'
        'hXdkP>RARr=cVQpn1Iv^rvZ*OERa%Ev;Z*FXPEpK9TWpZ|9aw04WARr(hB64MPb#iVZIv{%rARr(hARr=hV`yb<VPAG-'
        'a&u{KZX!A$F)RuoARr(hAR=;Qadl;LbYE#?B03-<V{dMAbaHiLbZKvHEpRn0Ffk%53LqdLARr(jaBN|2Uvp?-'
        'GBq|LIv^r3F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YBED9hXA'
        'Rr(hB4c55WnXDzB03-<P&F-Oc4cmKEpK9TWpZ|9A}k6ZARr(hAR=>gVRUtKB03-<P(f2uUqM4dMNm{lL?SE-ARr(hARr=Tc4'
        '=f~ZewL%b7)~QH8vtTAR;m{GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBP'
        'qUGBP483LqdLARr(jZe@6MUt?i&Wg<EtB2YCgWOZ<CX=7n@Wi4i9ZewL4ED9hXARr(hB6V(ZWpi(Ac4cHDIv{3YY;$ES3Lqd'
        'LARr(ja5Z0Zbz*X5b9HQVB03;@3LqdLARr(hAR=vHbYo~~Zf9R%aByFAa%pF0WpZC<bzyR3UuAe@a${+8Wn>~cAZB4~b7d?F'
        'ARr(hARr(hB5z`IWpZ|9WM6M)W?yn;W?yA?Wo~pLIv{R!Y-'
        '}tFARr(hARr(hB6DS9Z*F8?WO8YCWpZC+X>)L4bYo~=V{dhCbRs$+Ff0lnARr(hARr(jVQ^_*ZFO{EbZKvHUt@1|Zge6#ATT'
        'TnARr(hARr(hB4=rIUvPDEXkTM*b#8PbIv_AC3LqdLARr(hAR=XWX>)XGZf9R+Wo~0-Uvp?-B03-'
        '<Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YA}k6ZARr(hARr(jW@T<;WnXi2VRU68Iv^reMN&ydNkv3o'
        'O+i#cNFpo>ARr(hARr(hB64MMb!BsOUvO+;ZeML-bYo~DIv{j%b!99HARr(hARr(hB4%M}Y;|&FUvgz(b8l`UIv{R!Y-'
        '|c3ARr(hAbkoTARr)pED9hXARr=TcyMK7bY)~;c42I3WFk5sbaHiNED9hXARr=PV{~<4Y+rU^Y-'
        'wa7Iv{j%b!7@5Abl(fARv1RARr(hB5q-'
        'AWg<EtB5z`IWpZ|9axHUZV{dL`Uu1G=c4cy3WNCA7VRU0?Ut@1|ZgeefZ*F>Ja&IkTVQy}3bS-dUb8{jr3LqdLAR=;QbairW'
        'B03;@3LqdLARr(jb7N>_ZDC(_WpZ<AZ*C$wATcZoARr(hARr=gWpQ<7b97&6WFk5sB4clEb98cbV{~b6ZY^*%Eif@6ED9hXA'
        'Rr(hB5-VBZeMd~VKOx~B03-'
        '<F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQA}k6ZARr(hAR=R7'
        'b7fy?WFk5sB2YCgWp-t5bS-aUb7gXNWg;vJARr(hARr=hbYXOLb0Rt*B2Yn7Q(r+tLq$+jMMNSj3LqdLARr(jWp-&~Wo~0-'
        'Uvp?-'
        'GBq|LIv^r4GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqEED9hXA'
        'Rr(hB5q}PbYEj(b7dkrAR<sTEo60YY-wX*bY(4OWo~0-'
        'A}k6ZARr(hAR={ca%FRGY<6X2B03;uVQh0{ED9hXARr(hB5*Zdb9G{JWpj0GbRs$+dkP>RARr(hARr=bVRU0?X>Mm<VQ_F?b'
        'aH8DXJv9<XLVt6WM5@?WpZO_b7f>AIv{3YY;$ES3LqdLARr(hAR=#Kb7gXNWn^D(W@cY<WoBPxc4cmKB03;$b!==b3LqdLAR'
        'r(hAR=>RV{dL`Uu1G=c4cy3WNCA7VRU0?Ut@1|Zge6#ATcZoARr(hARr(hB4Kc8Uu|`CVRUJ4ZeL?>b#8PbIv_AC3LqdLARr'
        '(hAR=dJbYF0Fb7)^<Z*^{TB03;2ED9hXARr(hARr=TcxiKVX>Mm<W@T<;WnXh>VIn#pA~Q2HGcz+YGcz+YGcz+YGcz+YGcz+'
        'YGcz+YGcz+YGcz+YGcz+YGa@VsARr(hARr(hB4%Z7V`X1+bYXO5B03-'
        '<Rz*@tMoC3PUrj+&Lr5Yl3LqdLARr(hAR=;Qadl;LbYF06VQyb-VRU0?B03;+a&=`a3LqdLARr(hAR=aAX>4_JWnXe-VRLV8'
        'B03;$b!==3ARr(hARv7TARr(heJlzfARr(jWq5F9V{~O?Uv^<^X=EZgAZB4~b7d?FARr(hB4J~6bzy8@c42I3WFk5sW?^h|W'
        'eOl5eJlzfAbScRARr(jZeeX@B03-<Z(?(0a&~2MEpugKZ*F8?WO8YCWpZC+X>)L4bYo~=V{dhCbS-XgZhB>MZ!KbMZ)0m^WF'
        'jmIARr(hB64MPb#iVZIv{%rARr(hARr=hV`yb<VPAG-a&u{KZX!A$F)RuoARr(hAR=;Qadl;LbYE#?B03-'
        '<V{dMAbaHiLbZKvHEpRn0Ffk%53LqdLARr(jaBN|2Uvp?-'
        'GBq|LIv^r3F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YBED9hXA'
        'Rr(hB4c55WnXDzB03-<P&F-Oc4cmKEpK9TWpZ|9A}k6ZARr(hAR=>gVRUtKB03-'
        '<LQGFXOGQLqRZc}%P(?#jMMPggLqkPSR6$NdMItN;ARr(hARr=Tc4=f~ZewL%b7)~QH8vtTAR;m{GBPqUGBPqUGBPqUGBPqU'
        'GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBP483LqdLARr(jZe@6MUt?i&Wg<EtZgp&IED9hXA'
        'Rr(hB6V(ZWpi(Ac4cHDIv{j%b!99HARr(hARr=eHD7aeVsd43b!>DZIv{%rARr(hARr(hB5h%GV`yn^XJ27(a9?zCX=i6;a$'
        'je4VRB?&Wq4(BV`+0`WFk5sW?^h|Wh@FHARr(hARr(jZ(?(0a&~28UvFk+Uvgz;UuAY>Zge6#AZ~SRY%B^OARr(hARr(jb7f'
        '<1Ze(9%a%pyDa$jU=b8umFV`yJvZ*^{TB03;3ED9hXARr(hARr=PaA{v{b#!5LX>V>{V{dhCbRs$+Ff0lnARr(hARr(jXK8d'
        '_aCLKNUt@1|Zge6#ATTTnARr(hARr(hB4v1Kb98BLXJ2MzZewL%b7)~AIv^r5Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+Y'
        'Gcz+YGcz+YGcz+HED9hXARr(hARr=UWo~0-'
        'UvqR}bY&tsAR<;pQb|ThMMPgsK~zIXA}k6ZARr(hARr(ja%FLKWpi|2aBN|2Uu|J@V`w5eAartdWh@FHARr(hARr(jW?^Y;b'
        '#i52a%Ew2Z*C$wAR=#Kb7gXNWn?X7W@cq$bS-akEp={dZf|#PA_^cNARr(heF`8TARv7#3LqdLAR=XWaAjk3Wn^D=VQgt+B0'
        '3;+a&=`a3LqdLAR=L7bai2DUv^<^X=EZgAartdWeOl5eJlzfAbScRARr(jZeeX@B03-'
        '<Z(?(0a&~2MEpugKZ*F8?WO8YCWpZC+X>)L4bYo~=V{dhCbS-smYi@6MZY^VBZf<XMEpTCTb0RDXARr(hB64MPb#iVZIv{%r'
        'ARr(hARr=hV`yb<VPAG-a&u{KZX!A$F)RuoARr(hAR=;Qadl;LbYE#?B03-'
        '<V{dMAbaHiLbZKvHEpRn0Ffk%53LqdLARr(jaBN|2Uvp?-'
        'GBq|LIv^r3F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YBED9hXA'
        'Rr(hB4c55WnXDzB03-<P&F-Oc4cmKEpK9TWpZ|9A}k6ZARr(hAR=>gVRUtKB03-<P(f2uUqM4dMNm{lL?SE-ARr(hARr=Tc4'
        '=f~ZewL%b7)~QH8vtTAR;m{GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBP'
        'qUGBP483LqdLARr(jZe@6MUt?i&Wg<EtB2YCgWOZ<CX=7n@Wi4i9ZewL4ED9hXARr(hB6V(ZWpi(Ac4cHDIv{3YY;$ES3Lqd'
        'LARr(ja5Z0Zbz*X5b9HQVB03;@3LqdLARr(hAR=vHbYo~~Zf9R%aByFAa%pF0WpZC<bzyR3UuAe@a${+8Wn>~cAZB4~b7d?F'
        'ARr(hARr(hB5z`IWpZ|9WM6M)W?yn;W?yA?Wo~pLIv{R!Y-'
        '}tFARr(hARr(hB6DS9Z*F8?WO8YCWpZC+X>)L4bYo~=V{dhCbRs$+Zgp&IED9hXARr(hARr=PaA{v{b#!5LX>V>{V{dhCbRs'
        '$+Ff0lnARr(hARr(jXK8d_aCLKNUt@1|Zge6#ATTTnARr(hARr(hB4v1Kb98BLXJ2MzZewL%b7)~AIv^r5Gcz+YGcz+YGcz+'
        'YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+HED9hXARr(hARr=UWo~0-'
        'UvqR}bY&tsAR<;pQb|ThMMPgsK~zIXA}k6ZARr(hARr(ja%FLKWpi|2aBN|2Uu|J@V`w5eAartdWh@FHARr(hARr(jW?^Y;b'
        '#i52a%Ew2Z*C$wAZ~SRYziPCARr(heF`8TARv7#3LqdLAR=XWaAjk3Wn^D=VQgt+B03;uVQh0{ED9hXARr=PV{~<4Y+rU^Y-'
        'wa7Iv{3YY;$D_ARv7#3LqeR3LqdLAR=yIZDk@lAR=#Kb7gXNWpXWZWn*t{WM5=*X?A6DUu0=>aA9;~XkTM*b#8Pmb#7~JZ+C'
        '7jVr*|?Yh`32ED9hXARr=gWps6NZX!A$dkP>RARr(hB6DMCWo=<!c4cyNX>V>KIv_DD3LqdLARr(ja%FLKWpi|2X=EZgAR=R'
        'LZgX^Ubz^jCZ*DDcH7zhPA}k6ZARr(hAR=&VVQyb@XkjunHX=G8A~7*BF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQ'
        'F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF(NDqARr(hARr=RVRL0)X=EZgAR<sTEoF9PZgeegVsmA3c4Z<g3LqdLARr(jb'
        '97;Jb#o#*AR<CcPeV&ZL|;`-'
        'OHNN$P9iJ{ARr(hARr=Tc4=f~ZewL%b7)~QH8vtTAR;m{GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUG'
        'BPqUGBPqUGBPqUGBPqUGBPqUGBP483LqdLARr(jZe@6MUt?i&Wg<EtZgp&IED9hXARr(hB6V(ZWpi(Ac4cHDIv{j%b!99HAR'
        'r(hARr=eHD7aeVsd43b!>DZIv{%rARr(hARr(hB5h%GV`yn^XJ27(a9?zCX=i6;a$je4VRB?&Wq4(BV`+0`WFk5sW?^h|Wh@'
        'FHARr(hARr(jZ(?(0a&~28UvFk+Uvgz;UuAY>Zge6#AZ~SRY%B^OARr(hARr(jb7f<1Ze(9%a%pyDa$jU=b8umFV`yJvZ*^{'
        'TB03;$b!==b3LqdLARr(hAR=LKX<u!1bYXO9Z*E^>Z*^{TB03;2ED9hXARr(hARr=VX>?z3b#rK6V{dhCbRs$+Ff0lnARr(h'
        'ARr(jWq4_GbZKs9UuI=)V`X1+Xkj8cAR;p}Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGczJA3LqdLARr(hA'
        'R=aEZewL%b97;JWg<EtB34CGNk&OUL|;uoR6|H2ED9hXARr(hARr=gWpQ<7b97&DY+-'
        'I+ZDDj{Xd*fwbaHiNED9hXARr(hARr=UVQFl2a%Ep~Wnpt~ZX!A$B5z`IWpZ|9WG!W8W@Te^EpKuyb#7~JZ+C7Y3LqdLARr)'
        'p3LqdLAbl(fARr(hB4v1RWn*+@WM6h+Y-'
        'wa7Iv{j%b!99HARr(hB4J~6bzy8@c42I3WFk5sbaHiN3LqeTED9hXdkP>RARr=cVQpn1Iv^r%VsmA3c4cxcb7f<1Ze(9%a%p'
        'yDa$jU=b8umFV`yJvZ*^{TEn;tPY+-C^VRJ2GVQy}3bS-dUb8{jr3LqdLAR=;QbairWB03;@3LqdLARr(jb7N>_ZDC(_WpZ<'
        'AZ*C$wATcZoARr(hARr=gWpQ<7b97&6WFk5sB4clEb98cbV{~b6ZY^*%Eif@6ED9hXARr(hB5-VBZeMd~VKOx~B03-'
        '<F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQA}k6ZARr(hAR=R7'
        'b7fy?WFk5sB2YCgWp-t5bS-aUb7gXNWg;vJARr(hARr=hbYXOLb0Rt*B2Yn7Q(r+tLq$+jMMNSj3LqdLARr(jWp-&~Wo~0-'
        'Uvp?-'
        'GBq|LIv^r4GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqEED9hXA'
        'Rr(hB5q}PbYEj(b7dkrAR<sTEo60YY-wX*bY(4OWo~0-'
        'A}k6ZARr(hAR={ca%FRGY<6X2B03;uVQh0{ED9hXARr(hB5*Zdb9G{JWpj0GbRs$+dkP>RARr(hARr=bVRU0?X>Mm<VQ_F?b'
        'aH8DXJv9<XLVt6WM5@?WpZO_b7f>AIv{3YY;$ES3LqdLARr(hAR=#Kb7gXNWn^D(W@cY<WoBPxc4cmKB03;$b!==b3LqdLAR'
        'r(hAR=>RV{dL`Uu1G=c4cy3WNCA7VRU0?Ut@1|Zge6#AZB4~b7d?FARr(hARr(hB4Kc8Uu|`CVRUJ4ZeL?>b#8PbIv_AC3Lq'
        'dLARr(hAR=dJbYF0Fb7)^<Z*^{TB03;2ED9hXARr(hARr=TcxiKVX>Mm<W@T<;WnXh>VIn#pA~Q2HGcz+YGcz+YGcz+YGcz+'
        'YGcz+YGcz+YGcz+YGcz+YGcz+YGa@VsARr(hARr(hB4%Z7V`X1+bYXO5B03-'
        '<Rz*@tMoC3PUrj+&Lr5Yl3LqdLARr(hAR=;Qadl;LbYF06VQyb-VRU0?B03;+a&=`a3LqdLARr(hAR=aAX>4_JWnXe-VRLV8'
        'B03;$b!==3ARr(hARv7TARr(heJlzfARr(jWq5F9V{~O?Uv^<^X=EZgAZB4~b7d?FARr(hB4J~6bzy8@c42I3WFk5sW?^h|W'
        'eOl5eJlzfAbScRARr(jZeeX@B03-'
        '<Z(?(0a&~2MEpugKZ*F8?WO8YCWpZC+X>)L4bYo~=V{dhCbS+|UZ){;~X<>6MVr*|?Yh`32ED9hXARr=gWps6NZX!A$dkP>R'
        'ARr(hB6DMCWo=<!c4cyNX>V>KIv_DD3LqdLARr(ja%FLKWpi|2X=EZgAR=RLZgX^Ubz^jCZ*DDcH7zhPA}k6ZARr(hAR=&VV'
        'Qyb@XkjunHX=G8A~7*BF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF('
        'NDqARr(hARr=RVRL0)X=EZgAR<sTEoF9PZgeegVsmA3c4Z<g3LqdLARr(jb97;Jb#o#*AR<CcPeV&ZL|;`-'
        'MOaWpLsUgXUqM4dMNm{hPD4c^ED9hXARr(hB4u`IWMyt+WnXh>VKOx~B03-'
        '<GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUA}k6ZARr(hAR=yM'
        'cywQ5VRL07Iv{R!Y-'
        '}tFARr(hARr=jZgORFZ)|pDWFk5sbaHiNED9hXARr(hB5*Zdb9G{JWpj0GbRs$+dkP>RARr(hARr=bVRU0?X>Mm<VQ_F?baH'
        '8DXJv9<XLVt6WM5@?WpZO_b7f>AIv{3YY;$ES3LqdLARr(hAR=#Kb7gXNWn^D(W@cY<WoBPxc4cmKB03;$b!==b3LqdLARr('
        'hAR=>RV{dL`Uu1G=c4cy3WNCA7VRU0?Ut@1|Zge6#AZB4~b7d?FARr(hARr(hB4Kc8Uu|`CVRUJ4ZeL?>b#8PbIv_AC3LqdL'
        'ARr(hAR=dJbYF0Fb7)^<Z*^{TB03;2ED9hXARr(hARr=TcxiKVX>Mm<W@T<;WnXh>VIn#pA~Q2HGcz+YGcz+YGcz+YGcz+YG'
        'cz+YGcz+YGcz+YGcz+YGcz+YGa@VsARr(hARr(hB4%Z7V`X1+bYXO5B03-'
        '<Rz*@tMoC3PUrj+&Lr5Yl3LqdLARr(hAR=;Qadl;LbYF06VQyb-VRU0?B03;+a&=`a3LqdLARr(hAR=aAX>4_JWnXe-VRLV8'
        'B03-'
        '<Z(?(0a&~28EoEkAWn*+LZ*nblZfkCDcWxpIARr(hARv7TARr(heJlzfARr(jWq5F9V{~O?Uv^<^X=EZgAZB4~b7d?FARr(h'
        'B4J~6bzy8@c42I3WFk5sW?^h|WeOl5eJlzfAbScRARr(jZeeX@B03-'
        '<Z(?(0a&~2MEn#qJUu|`CVRUJ4ZeL?>b#8PmZf|aSWpZyVV_|M?Z*(niVRLgLED9hXARr=gWps6NZX!A$dkP>RARr(hB6DMC'
        'Wo=<!c4cyNX>V>KIv_DD3LqdLARr(ja%FLKWpi|2X=EZgAR=RLZgX^Ubz^jCZ*DDcH7zhPA}k6ZARr(hAR=&VVQyb@XkjunH'
        'X=G8A~7*BF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF(NDqARr(hAR'
        'r=RVRL0)X=EZgAR<sTEoF9PZgeegVsmA3c4Z<g3LqdLARr(jb97;Jb#o#*AR<sfQ&V3-'
        'LqkPSR7FH0ED9hXARr(hB4u`IWMyt+WnXh>VKOx~B03-'
        '<GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUA}k6ZARr(hAR=yM'
        'cywQ5VRL07Iv^rYH7#UyaBOK~VRU6JW@T<;Wg;vJARr(hARr=jZgORFZ)|pDWFk5sW?^h|Wh@FHARr(hAR=%zUvqV0a%FRMY'
        ';+<zAbScRARr(hARr(jZDDj{XlZU|Utw@?UvzS5XJ=({UuSh;a%5j+cx7^9X>(;{B03;uVQh0{ED9hXARr(hARr=dVsmA3c4'
        'cH=Z)Rp+a%E;;Wp-'
        't5bRs$+Zgp&IED9hXARr(hARr=hWn*t{WM5=*X?A6DUu0=>aA9;~XkTM*b#8PbIv_AC3LqdLARr(hAR=LKX<u!1bYXO9Z*E^'
        '>Z*^{TB03;3ED9hXARr(hARr=VX>?z3b#rK6V{dhCbRs$+Ff0lnARr(hARr(jWq4_GbZKs9UuI=)V`X1+Xkj8cAR;p}Gcz+Y'
        'Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGczJA3LqdLARr(hAR=aEZewL%b97;JWg<EtB34CGNk&OUL|;uoR6|H2E'
        'D9hXARr(hARr=gWpQ<7b97&DY+-'
        'I+ZDDj{Xd*fwbaHiNED9hXARr(hARr=UVQFl2a%Ep~Wnpt~ZX!A$Zgp&I3LqdLARr)p3LqdLAbl(fARr(hB4v1RWn*+@WM6h'
        '+Y-wa7Iv{3YY;$ES3LqdLAR=L7bai2DUv^<^X=EZgAZB4~b7cx3Abl(fARv1RARr(hB5q-'
        'AWg<EtB5z`IWpZ|9axGzSX<u!1bYXO9Z*E^>Z*^{TEpBgadS!BNEn;kMV{2t(A}k6ZARr(ja%FUNa&96zAbScRARr(hAR=>P'
        'Xk~3-Uv_13b7^mGB03;3ED9hXARr(hB64MMb!BsOUuk3_Iv^rrZ*FsRa&=>LX>V>Va5XJ3F(NDqARr(hARr=eY+-'
        'I+b7)~QH8vtTAR;j_F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)<='
        '63LqdLARr(jV_|b;Uuk3_Iv^rYH7#X!Wo~pWZ(?(0a&~1RED9hXARr(hB6D<MbaitgIv^rKOix2gMMPg!PDNNyMMG3YL|;Ke'
        'Lq$+jK~6(OA}k6ZARr(hAR=XUX=G(?V`X1+XkjunHX=G8A~G^EGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUG'
        'BPqUGBPqUGBPqUGBPqUGBPqUGBPqUG9oMrARr(hARr=cWq5R7V_|b;B03;$b!==b3LqdLARr(jb#8KHb8l>RWn>~cAartdWh'
        '@FHARr(hAR=%zUvqV0a%FRMY;+<zAbScRARr(hARr(jZDDj{XlZU|Utw@?UvzS5XJ=({UuSh;a%5j+cx7^9X>(;{B03;uVQh'
        '0{ED9hXARr(hARr=dVsmA3c4cH=Z)Rp+a%E;;Wp-'
        't5bRs$+Zgp&IED9hXARr(hARr=hWn*t{WM5=*X?A6DUu0=>aA9;~XkTM*b#8PbIv_AC3LqdLARr(hAR=LKX<u!1bYXO9Z*E^'
        '>Z*^{TB03;3ED9hXARr(hARr=VX>?z3b#rK6V{dhCbRs$+Ff0lnARr(hARr(jWq4_GbZKs9UuI=)V`X1+Xkj8cAR;p}Gcz+Y'
        'Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGczJA3LqdLARr(hAR=aEZewL%b97;JWg<EtB34CGNk&OUL|;uoR6|H2E'
        'D9hXARr(hARr=gWpQ<7b97&DY+-'
        'I+ZDDj{Xd*fwbaHiNED9hXARr(hARr=UVQFl2a%Ep~Wnpt~ZX!A$B5z`IWpZ|9WG!W8W@Te^EpKuyb#7~JZ+C7Y3LqdLARr)'
        'p3LqdLAbl(fARr(hB4v1RWn*+@WM6h+Y-'
        'wa7Iv{j%b!99HARr(hB4J~6bzy8@c42I3WFk5sbaHiN3LqeTED9hXdkP>RARr=cVQpn1Iv^r%VsmA3c4cxcVQ^_*ZFO{EbZK'
        'vHUt@1|ZgeemZfkCDcWy0XVQy}3bS-dUb8{jr3LqdLAR=;QbairWB03;@3LqdLARr(jb7N>_ZDC(_WpZ<AZ*C$wATcZoARr('
        'hARr=gWpQ<7b97&6WFk5sB4clEb98cbV{~b6ZY^*%Eif@6ED9hXARr(hB5-VBZeMd~VKOx~B03-'
        '<F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQA}k6ZARr(hAR=R7'
        'b7fy?WFk5sB2YCgWp-t5bS-aUb7gXNWg;vJARr(hARr=hbYXOLb0Rt*B2Yn7Q(r+tLq$+jMMNSj3LqdLARr(jWp-&~Wo~0-'
        'Uvp?-'
        'GBq|LIv^r4GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqEED9hXA'
        'Rr(hB5q}PbYEj(b7dkrAR<sTEo60YY-wX*bY(4OWo~0-'
        'A}k6ZARr(hAR={ca%FRGY<6X2B03;uVQh0{ED9hXARr(hB5*Zdb9G{JWpj0GbRs$+dkP>RARr(hARr=bVRU0?X>Mm<VQ_F?b'
        'aH8DXJv9<XLVt6WM5@?WpZO_b7f>AIv{3YY;$ES3LqdLARr(hAR=#Kb7gXNWn^D(W@cY<WoBPxc4cmKB03;$b!==b3LqdLAR'
        'r(hAR=>RV{dL`Uu1G=c4cy3WNCA7VRU0?Ut@1|Zge6#ATTTnARr(hARr(hB4Kc8Uu|`CVRUJ4ZeL?>b#8PbIv{R!Y-'
        '}tFARr(hARr(hB4=rIUvPDEXkTM*b#8PbIv_AC3LqdLARr(hAR=XWX>)XGZf9R+Wo~0-Uvp?-B03-'
        '<Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YA}k6ZARr(hARr(jW@T<;WnXi2VRU68Iv^reMN&ydNkv3o'
        'O+i#cNFpo>ARr(hARr(hB64MMb!BsOUvO+;ZeML-bYo~DIv{j%b!99HARr(hARr(hB4%M}Y;|&FUvgz(b8l`UIv{R!Y-'
        '|c3ARr(hAbkoTARr)pED9hXARr=TcyMK7bY)~;c42I3WFk5sW?^h|Wh@FHARr(jVPkZ4VQgP^VQgt+B03;uVQh0{3LqeTED9'
        'hXdkP>RARr=cVQpn1Iv^r%VsmA3c4cxcVQ^_*ZFO{EbZKvHUt@1|ZgeemZfkCDcWy0WY;R+0Wn>~O3LqdLAR=;QbairWB03;'
        '@3LqdLARr(jb7N>_ZDC(_WpZ<AZ*C$wATcZoARr(hARr=gWpQ<7b97&6WFk5sB4clEb98cbV{~b6ZY^*%Eif@6ED9hXARr(h'
        'B5-VBZeMd~VKOx~B03-'
        '<F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQA}k6ZARr(hAR=R7'
        'b7fy?WFk5sB2YCgWp-t5bS-aUb7gXNWg;vJARr(hARr=hbYXOLb0Rt*B0@}0LrX<OUsX;^PES`(A}k6ZARr(hAR=XUX=G(?V'
        '`X1+XkjunHX=G8A~G^EGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUG9'
        'oMrARr(hARr=cWq5R7V_|b;B03;$b!==b3LqdLARr(jb#8KHb8l>RWn>~cAartdWh@FHARr(hAR=%zUvqV0a%FRMY;+<zAbS'
        'cRARr(hARr(jZDDj{XlZU|Utw@?UvzS5XJ=({UuSh;a%5j+cx7^9X>(;{B03;uVQh0{ED9hXARr(hARr=dVsmA3c4cH=Z)Rp'
        '+a%E;;Wp-'
        't5bRs$+Zgp&IED9hXARr(hARr=hWn*t{WM5=*X?A6DUu0=>aA9;~XkTM*b#8PbIv_AC3LqdLARr(hAR=LKX<u!1bYXO9Z*E^'
        '>Z*^{TB03;$b!==b3LqdLARr(hAR=dJbYF0Fb7)^<Z*^{TB03;2ED9hXARr(hARr=TcxiKVX>Mm<W@T<;WnXh>VIn#pA~Q2H'
        'Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGa@VsARr(hARr(hB4%Z7V`X1+bYXO5B03-'
        '<Rz*@tMoC3PUrj+&Lr5Yl3LqdLARr(hAR=;Qadl;LbYF06VQyb-VRU0?B03;+a&=`a3LqdLARr(hAR=aAX>4_JWnXe-VRLV8'
        'B03-'
        '<Z(?(0a&~28EoEkAWn*+LZ*nblZfkCDcWxpIARr(hARv7TARr(heJlzfARr(jWq5F9V{~O?Uv^<^X=EZgAartdWh@FHARr(j'
        'VPkZ4VQgP^VQgt+B03;+a&=`2ARv7#3LqeR3LqdLAR=yIZDk@lAR=#Kb7gXNWpXWHaA{v{b#!5LX>V>{V{dhCbS+|UZ){;~X'
        '<>6MV_|M?Z*(niVRLgLED9hXARr=gWps6NZX!A$dkP>RARr(hB6DMCWo=<!c4cyNX>V>KIv_DD3LqdLARr(ja%FLKWpi|2X='
        'EZgAR=RLZgX^Ubz^jCZ*DDcH7zhPA}k6ZARr(hAR=&VVQyb@XkjunHX=G8A~7*BF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)='
        'YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF(NDqARr(hARr=RVRL0)X=EZgAR<sTEoF9PZgeegVsmA3c4Z<g3Lqd'
        'LARr(jb97;Jb#o#*AR<sfQ&V3-LqkPSR7FH0ED9hXARr(hB4u`IWMyt+WnXh>VKOx~B03-'
        '<GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUA}k6ZARr(hAR=yM'
        'cywQ5VRL07Iv^rYH7#UyaBOK~VRU6JW@T<;Wg;vJARr(hARr=jZgORFZ)|pDWFk5sW?^h|Wh@FHARr(hAR=%zUvqV0a%FRMY'
        ';+<zAbScRARr(hARr(jZDDj{XlZU|Utw@?UvzS5XJ=({UuSh;a%5j+cx7^9X>(;{B03;uVQh0{ED9hXARr(hARr=dVsmA3c4'
        'cH=Z)Rp+a%E;;Wp-'
        't5bRs$+Zgp&IED9hXARr(hARr=hWn*t{WM5=*X?A6DUu0=>aA9;~XkTM*b#8PbIv_AC3LqdLARr(hAR=LKX<u!1bYXO9Z*E^'
        '>Z*^{TB03;uVQh0{ED9hXARr(hARr=VX>?z3b#rK6V{dhCbRs$+Ff0lnARr(hARr(jWq4_GbZKs9UuI=)V`X1+Xkj8cAR;p}'
        'Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGczJA3LqdLARr(hAR=aEZewL%b97;JWg<EtB34CGNk&OUL|;uoR'
        '6|H2ED9hXARr(hARr=gWpQ<7b97&DY+-'
        'I+ZDDj{Xd*fwbaHiNED9hXARr(hARr=UVQFl2a%Ep~Wnpt~ZX!A$Zgp&I3LqdLARr)p3LqdLAbl(fARr(hB4v1RWn*+@WM6h'
        '+Y-wa7Iv{3YY;$ES3LqdLAR=L7bai2DUv^<^X=EZgAZB4~b7cx3Abl(fARv1RARr(hB5q-'
        'AWg<EtB5z`IWpZ|9axGzSX<u!1bYXO9Z*E^>Z*^{TEn;tPY+-'
        'C^VRJ2FY;R+0Wn>~O3LqdLAR=;QbairWB03;@3LqdLARr(jb7N>_ZDC(_WpZ<AZ*C$wATcZoARr(hARr=gWpQ<7b97&6WFk5'
        'sB4clEb98cbV{~b6ZY^*%Eif@6ED9hXARr(hB5-VBZeMd~VKOx~B03-'
        '<F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQA}k6ZARr(hAR=R7'
        'b7fy?WFk5sB2YCgWp-t5bS-aUb7gXNWg;vJARr(hARr=hbYXOLb0Rt*B0@}0LrX<OUsX;;SWrbnR7FHzK|@1DP*g!qLq#Gi3'
        'LqdLARr(jWp-&~Wo~0-Uvp?-'
        'GBq|LIv^r4GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqEED9hXA'
        'Rr(hB5q}PbYEj(b7dkrAZ~SRY%B^OARr(hAR={ca%FRGY<6X2B03;+a&=`a3LqdLARr(ja5Z0Zbz*X5b9HQVB03;@3LqdLAR'
        'r(hAR=vHbYo~~Zf9R%aByFAa%pF0WpZC<bzyR3UuAe@a${+8Wn>~cAZB4~b7d?FARr(hARr(hB5z`IWpZ|9WM6M)W?yn;W?y'
        'A?Wo~pLIv{R!Y-'
        '}tFARr(hARr(hB6DS9Z*F8?WO8YCWpZC+X>)L4bYo~=V{dhCbRs$+Ff0lnARr(hARr(jVQ^_*ZFO{EbZKvHUt@1|Zge6#AZB'
        '4~b7d?FARr(hARr(hB4=rIUvPDEXkTM*b#8PbIv_AC3LqdLARr(hAR=XWX>)XGZf9R+Wo~0-Uvp?-B03-'
        '<Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YA}k6ZARr(hARr(jW@T<;WnXi2VRU68Iv^reMN&ydNkv3o'
        'O+i#cNFpo>ARr(hARr(hB64MMb!BsOUvO+;ZeML-bYo~DIv{j%b!99HARr(hARr(hB4%M}Y;|&FUvgz(b8l`UIv^r%VsmA3c'
        '4cHOWoBk&V{|QVaxHakYi@6MZXyaGARr(hAbkoTARr)pED9hXARr=TcyMK7bY)~;c42I3WFk5sW?^h|Wh@FHARr(jVPkZ4VQ'
        'gP^VQgt+B03;uVQh0{3LqeTED9hXdkP>RARr=cVQpn1Iv^r%VsmA3c4cxcXK8d_aCLKNUt@1|ZgeefZ*F>Ja&IkTVQy}3bS-'
        'dUb8{jr3LqdLAR=;QbairWB03;@3LqdLARr(jb7N>_ZDC(_WpZ<AZ*C$wATcZoARr(hARr=gWpQ<7b97&6WFk5sB4clEb98c'
        'bV{~b6ZY^*%Eif@6ED9hXARr(hB5-VBZeMd~VKOx~B03-'
        '<F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQA}k6ZARr(hAR=R7'
        'b7fy?WFk5sB2YCgWp-t5bS-aUb7gXNWg;vJARr(hARr=hbYXOLb0Rt*B2Yn7Q(r+tLq$+jMMNSj3LqdLARr(jWp-&~Wo~0-'
        'Uvp?-'
        'GBq|LIv^r4GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqEED9hXA'
        'Rr(hB5q}PbYEj(b7dkrAR<sTEo60YY-wX*bY(4OWo~0-'
        'A}k6ZARr(hAR={ca%FRGY<6X2B03;uVQh0{ED9hXARr(hB5*Zdb9G{JWpj0GbRs$+dkP>RARr(hARr=bVRU0?X>Mm<VQ_F?b'
        'aH8DXJv9<XLVt6WM5@?WpZO_b7f>AIv{3YY;$ES3LqdLARr(hAR=#Kb7gXNWn^D(W@cY<WoBPxc4cmKB03;$b!==b3LqdLAR'
        'r(hAR=>RV{dL`Uu1G=c4cy3WNCA7VRU0?Ut@1|Zge6#ATTTnARr(hARr(hB4Kc8Uu|`CVRUJ4ZeL?>b#8PbIv_AC3LqdLARr'
        '(hAR=dJbYF0Fb7)^<Z*^{TB03;3ED9hXARr(hARr=TcxiKVX>Mm<W@T<;WnXh>VIn#pA~Q2HGcz+YGcz+YGcz+YGcz+YGcz+'
        'YGcz+YGcz+YGcz+YGcz+YGa@VsARr(hARr(hB4%Z7V`X1+bYXO5B03-'
        '<Rz*@tMoC3PUrj+&Lr5Yl3LqdLARr(hAR=;Qadl;LbYF06VQyb-VRU0?B03;+a&=`a3LqdLARr(hAR=aAX>4_JWnXe-VRLV8'
        'B03;$b!==3ARr(hARv7TARr(heJlzfARr(jWq5F9V{~O?Uv^<^X=EZgAZB4~b7d?FARr(hB4J~6bzy8@c42I3WFk5sW?^h|W'
        'eOl5eJlzfAbScRARr(jZeeX@B03-'
        '<Z(?(0a&~2MEoW(TUvPDEXkTM*b#8PmZf|aSWpZyVVr*|?Yh`32ED9hXARr=gWps6NZX!A$dkP>RARr(hB6DMCWo=<!c4cyN'
        'X>V>KIv_DD3LqdLARr(ja%FLKWpi|2X=EZgAR=RLZgX^Ubz^jCZ*DDcH7zhPA}k6ZARr(hAR=&VVQyb@XkjunHX=G8A~7*BF'
        ')=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF(NDqARr(hARr=RVRL0)X='
        'EZgAR<sTEoF9PZgeegVsmA3c4Z<g3LqdLARr(jb97;Jb#o#*AR<CcPeV&ZL|;`-'
        'MOaWpLsUgXUqM4dMNm{hPD4c^ED9hXARr(hB4u`IWMyt+WnXh>VKOx~B03-'
        '<GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUA}k6ZARr(hAR=yM'
        'cywQ5VRL07Iv{R!Y-'
        '}tFARr(hARr=jZgORFZ)|pDWFk5sbaHiNED9hXARr(hB5*Zdb9G{JWpj0GbRs$+dkP>RARr(hARr=bVRU0?X>Mm<VQ_F?baH'
        '8DXJv9<XLVt6WM5@?WpZO_b7f>AIv{3YY;$ES3LqdLARr(hAR=#Kb7gXNWn^D(W@cY<WoBPxc4cmKB03;$b!==b3LqdLARr('
        'hAR=>RV{dL`Uu1G=c4cy3WNCA7VRU0?Ut@1|Zge6#ATTTnARr(hARr(hB4Kc8Uu|`CVRUJ4ZeL?>b#8PbIv_AC3LqdLARr(h'
        'AR=dJbYF0Fb7)^<Z*^{TB03;3ED9hXARr(hARr=TcxiKVX>Mm<W@T<;WnXh>VIn#pA~Q2HGcz+YGcz+YGcz+YGcz+YGcz+YG'
        'cz+YGcz+YGcz+YGcz+YGa@VsARr(hARr(hB4%Z7V`X1+bYXO5B03-'
        '<Rz*@tMoC3PUrj+&Lr5Yl3LqdLARr(hAR=;Qadl;LbYF06VQyb-VRU0?B03;+a&=`a3LqdLARr(hAR=aAX>4_JWnXe-VRLV8'
        'B03-'
        '<Z(?(0a&~28EoEkAWn*+LZ*nblZfkCDcWxpIARr(hARv7TARr(heJlzfARr(jWq5F9V{~O?Uv^<^X=EZgAartdWh@FHARr(j'
        'VPkZ4VQgP^VQgt+B03;+a&=`2ARv7#3LqeR3LqdLAR=yIZDk@lAR=#Kb7gXNWpXWNX>?z3b#rK6V{dhCbS-smYi@6MZY^VBZ'
        'f<XMEpTCTb0RDXARr(hB64MPb#iVZIv{%rARr(hARr=hV`yb<VPAG-a&u{KZX!A$F)RuoARr(hAR=;Qadl;LbYE#?B03-'
        '<V{dMAbaHiLbZKvHEpRn0Ffk%53LqdLARr(jaBN|2Uvp?-'
        'GBq|LIv^r3F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YBED9hXA'
        'Rr(hB4c55WnXDzB03-<P&F-Oc4cmKEpK9TWpZ|9A}k6ZARr(hAR=>gVRUtKB03-<P(f2uUqM4dMNm{lL?SE-ARr(hARr=Tc4'
        '=f~ZewL%b7)~QH8vtTAR;m{GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBP'
        'qUGBP483LqdLARr(jZe@6MUt?i&Wg<EtB2YCgWOZ<CX=7n@Wi4i9ZewL4ED9hXARr(hB6V(ZWpi(Ac4cHDIv{3YY;$ES3Lqd'
        'LARr(ja5Z0Zbz*X5b9HQVB03;@3LqdLARr(hAR=vHbYo~~Zf9R%aByFAa%pF0WpZC<bzyR3UuAe@a${+8Wn>~cAZB4~b7d?F'
        'ARr(hARr(hB5z`IWpZ|9WM6M)W?yn;W?yA?Wo~pLIv{R!Y-'
        '}tFARr(hARr(hB6DS9Z*F8?WO8YCWpZC+X>)L4bYo~=V{dhCbRs$+Ff0lnARr(hARr(jVQ^_*ZFO{EbZKvHUt@1|Zge6#ATT'
        'TnARr(hARr(hB4=rIUvPDEXkTM*b#8PbIv{R!Y-'
        '}tFARr(hARr(hB4v1Kb98BLXJ2MzZewL%b7)~AIv^r5Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+HED9'
        'hXARr(hARr=UWo~0-'
        'UvqR}bY&tsAR<;pQb|ThMMPgsK~zIXA}k6ZARr(hARr(ja%FLKWpi|2aBN|2Uu|J@V`w5eAartdWh@FHARr(hARr(jW?^Y;b'
        '#i52a%Ew2Z*C$wAZ~SRYziPCARr(heF`8TARv7#3LqdLAR=XWaAjk3Wn^D=VQgt+B03;uVQh0{ED9hXARr=PV{~<4Y+rU^Y-'
        'wa7Iv{3YY;$D_ARv7#3LqeR3LqdLAR=yIZDk@lAR=#Kb7gXNWpXWNX>?z3b#rK6V{dhCbS-smYi@6MZY^SLZ)0m^WFjmIARr'
        '(hB64MPb#iVZIv{%rARr(hARr=hV`yb<VPAG-a&u{KZX!A$F)RuoARr(hAR=;Qadl;LbYE#?B03-'
        '<V{dMAbaHiLbZKvHEpRn0Ffk%53LqdLARr(jaBN|2Uvp?-'
        'GBq|LIv^r3F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YBED9hXA'
        'Rr(hB4c55WnXDzB03-<P&F-Oc4cmKEpK9TWpZ|9A}k6ZARr(hAR=>gVRUtKB03-'
        '<LQGFXOGQLqRZdGzPghPNED9hXARr(hB4u`IWMyt+WnXh>VKOx~B03-'
        '<GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUA}k6ZARr(hAR=yM'
        'cywQ5VRL07Iv{R!Y-'
        '}tFARr(hARr=jZgORFZ)|pDWFk5sbaHiNED9hXARr(hB5*Zdb9G{JWpj0GbRs$+dkP>RARr(hARr=bVRU0?X>Mm<VQ_F?baH'
        '8DXJv9<XLVt6WM5@?WpZO_b7f>AIv{3YY;$ES3LqdLARr(hAR=#Kb7gXNWn^D(W@cY<WoBPxc4cmKB03;$b!==b3LqdLARr('
        'hAR=>RV{dL`Uu1G=c4cy3WNCA7VRU0?Ut@1|Zge6#ATTTnARr(hARr(hB4Kc8Uu|`CVRUJ4ZeL?>b#8PbIv_AC3LqdLARr(h'
        'AR=dJbYF0Fb7)^<Z*^{TB03;$b!==b3LqdLARr(hAR=XWX>)XGZf9R+Wo~0-Uvp?-B03-'
        '<Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YGcz+YA}k6ZARr(hARr(jW@T<;WnXi2VRU68Iv^reMN&ydNkv3o'
        'O+i#cNFpo>ARr(hARr(hB64MMb!BsOUvO+;ZeML-bYo~DIv{j%b!99HARr(hARr(hB4%M}Y;|&FUvgz(b8l`UIv^r%VsmA3c'
        '4cHOWoBk&V{|QVaxHakYi@6MZXyaGARr(hAbkoTARr)pED9hXARr=TcyMK7bY)~;c42I3WFk5sbaHiNED9hXARr=PV{~<4Y+'
        'rU^Y-wa7Iv{j%b!7@5Abl(fARv1RARr(hB5q-AWg<EtB5z`IWpZ|9axG_RbYF0Fb7)^<Z*^{TEn;tPY+-C^VRJ2GVQy}3bS-'
        'dUb8{jr3LqdLAR=;QbairWB03;@3LqdLARr(jb7N>_ZDC(_WpZ<AZ*C$wATcZoARr(hARr=gWpQ<7b97&6WFk5sB4clEb98c'
        'bV{~b6ZY^*%Eif@6ED9hXARr(hB5-VBZeMd~VKOx~B03-'
        '<F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQA}k6ZARr(hAR=R7'
        'b7fy?WFk5sB2YCgWp-t5bS-aUb7gXNWg;vJARr(hARr=hbYXOLb0Rt*B2Yn7Q(r+tLq$+jMMNSj3LqdLARr(jWp-&~Wo~0-'
        'Uvp?-'
        'GBq|LIv^r4GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqEED9hXA'
        'Rr(hB5q}PbYEj(b7dkrAR<sTEo60YY-wX*bY(4OWo~0-'
        'A}k6ZARr(hAR={ca%FRGY<6X2B03;uVQh0{ED9hXARr(hB5*Zdb9G{JWpj0GbRs$+dkP>RARr(hARr=bVRU0?X>Mm<VQ_F?b'
        'aH8DXJv9<XLVt6WM5@?WpZO_b7f>AIv{3YY;$ES3LqdLARr(hAR=#Kb7gXNWn^D(W@cY<WoBPxc4cmKB03;$b!==b3LqdLAR'
        'r(hAR=>RV{dL`Uu1G=c4cy3WNCA7VRU0?Ut@1|Zge6#ATTTnARr(hARr(hB4Kc8Uu|`CVRUJ4ZeL?>b#8PbIv_AC3LqdLARr'
        '(hAR=dJbYF0Fb7)^<Z*^{TB03;uVQh0{ED9hXARr(hARr=TcxiKVX>Mm<W@T<;WnXh>VIn#pA~Q2HGcz+YGcz+YGcz+YGcz+'
        'YGcz+YGcz+YGcz+YGcz+YGcz+YGa@VsARr(hARr(hB4%Z7V`X1+bYXO5B03-'
        '<Rz*@tMoC3PUrj+&Lr5Yl3LqdLARr(hAR=;Qadl;LbYF06VQyb-VRU0?B03;+a&=`a3LqdLARr(hAR=aAX>4_JWnXe-VRLV8'
        'B03;$b!==3ARr(hARv7TARr(heJlzfARr(jWq5F9V{~O?Uv^<^X=EZgAZB4~b7d?FARr(hB4J~6bzy8@c42I3WFk5sW?^h|W'
        'eOl5eJlzfAbScRARr(jZeeX@B03-'
        '<Z(?(0a&~2MEoW(TUvPDEXkTM*b#8PmVsCG3VQgt(b1h<QZ)0m^WFjmIARr(hB64MPb#iVZIv{%rARr(hARr=hV`yb<VPAG-'
        'a&u{KZX!A$F)RuoARr(hAR=;Qadl;LbYE#?B03-<V{dMAbaHiLbZKvHEpRn0Ffk%53LqdLARr(jaBN|2Uvp?-'
        'GBq|LIv^r3F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YBED9hXA'
        'Rr(hB4c55WnXDzB03-<P&F-Oc4cmKEpK9TWpZ|9A}k6ZARr(hAR=>gVRUtKB03-'
        '<LQGFXOGQLqRZc}%P(?#jMMPggLqkPSR6$NdMItN;ARr(hARr=Tc4=f~ZewL%b7)~QH8vtTAR;m{GBPqUGBPqUGBPqUGBPqU'
        'GBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBP483LqdLARr(jZe@6MUt?i&Wg<EtZgp&IED9hXA'
        'Rr(hB6V(ZWpi(Ac4cHDIv{j%b!99HARr(hARr=eHD7aeVsd43b!>DZIv{%rARr(hARr(hB5h%GV`yn^XJ27(a9?zCX=i6;a$'
        'je4VRB?&Wq4(BV`+0`WFk5sW?^h|Wh@FHARr(hARr(jZ(?(0a&~28UvFk+Uvgz;UuAY>Zge6#AZ~SRY%B^OARr(hARr(jb7f'
        '<1Ze(9%a%pyDa$jU=b8umFV`yJvZ*^{TB03;2ED9hXARr(hARr=PaA{v{b#!5LX>V>{V{dhCbRs$+Ff0lnARr(hARr(jXK8d'
        '_aCLKNUt@1|Zge6#AZB4~b7d?FARr(hARr(hB4v1Kb98BLXJ2MzZewL%b7)~AIv^r5Gcz+YGcz+YGcz+YGcz+YGcz+YGcz+Y'
        'Gcz+YGcz+YGcz+YGcz+HED9hXARr(hARr=UWo~0-'
        'UvqR}bY&tsAR<;pQb|ThMMPgsK~zIXA}k6ZARr(hARr(ja%FLKWpi|2aBN|2Uu|J@V`w5eAartdWh@FHARr(hARr(jW?^Y;b'
        '#i52a%Ew2Z*C$wAR=#Kb7gXNWn?X7W@cq$bS-akEp={dZf|#PA_^cNARr(heF`8TARv7#3LqdLAR=XWaAjk3Wn^D=VQgt+B0'
        '3;uVQh0{ED9hXARr=PV{~<4Y+rU^Y-'
        'wa7Iv{3YY;$D_ARv7#3LqeR3LqdLAR=yIZDk@lAR=#Kb7gXNWpXWPZgyd8X=E*7WNKk!Wo~16A}k6ZARr(ja%FUNa&96zAbS'
        'cRARr(hAR=>PXk~3-'
        'Uv_13b7^mGB03;3ED9hXARr(hB64MMb!BsOUuk3_Iv^rrZ*FsRa&=>LX>V>Va5XJ3F(NDqARr(hARr=eY+-'
        'I+b7)~QH8vtTAR;j_F)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)=YQF)<='
        '63LqdLARr(jV_|b;Uuk3_Iv^rYH7#X!Wo~pWZ(?(0a&~1RED9hXARr(hB6D<MbaitgIv^rYK~qy-'
        'K|@1DP*g=kA}k6ZARr(hAR=XUX=G(?V`X1+XkjunHX=G8A~G^EGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUGBPqUG'
        'BPqUGBPqUGBPqUGBPqUGBPqUGBPqUG9oMrARr(hARr=cWq5R7V_|b;B03-<G-'
        'o0#3LqdLARr(jb#8KHb8l>RWn>~cAZB4~b7d?FARr(hARr=eHD7aeVsd43b!>DZIv{%rARr(hARr(hB5h%GV`yn^XJ27(a9?'
        'zCX=i6;a$je4VRB?&Wq4(BV`+0`WFk5sW?^h|Wh@FHARr(hARr(jZ(?(0a&~28UvFk+Uvgz;UuAY>Zge6#AZ~SRY%B^OARr('
        'hARr(jb7f<1Ze(9%a%pyDa$jU=b8umFV`yJvZ*^{TB03;2ED9hXARr(hARr=PaA{v{b#!5LX>V>{V{dhCbRs$+Ff0lnARr(h'
        'ARr(jXK8d_aCLKNUt@1|Zge6#ATTTnARr(hARr(hB4v1Kb98BLXJ2MzZewL%b7)~AIv^r5Gcz+YGcz+YGcz+YGcz+YGcz+YG'
        'cz+YGcz+YGcz+YGcz+YGcz+HED9hXARr(hARr=UWo~0-'
        'UvqR}bY&tsAR<;pQb|ThMMPgsK~zIXA}k6ZARr(hARr(ja%FLKWpi|2aBN|2Uu|J@V`w5eAartdWh@FHARr(hARr(jW?^Y;b'
        '#i52a%Ew2Z*C$wAZ~SRYziPCARr(heF`8TARv7#3LqdLAR=XWaAjk3Wn^D=VQgt+B03;uVQh0{ED9hXARr=PV{~<4Y+rU^Y-'
        'wa7Iv{3YY;$D_ARv7TT?z'
    ),
    'approval_marker_schema': (
        'dkP>RB6N9hWg<EtB5z`9Wn*+AED9hXB5-'
        'nVaAk6IX=QUFIv{%rARr(hB6DMCWo=<!c4cyNX>V>KIv{%rARr(hARr=RZ*FsRB03;33LqdLAbl(fARr(hB64MLZ*ysMZ*qA'
        'eIv{%rARr(hARr=RZ*FsRB03-'
        '<N_A{$VJ&TCWpr~bQ*?4^Zf7l2Xk~A5c`rj*L3n9)WpZ<6E^2Hd3LqdLAbl(fARr(hB64+ZUuk3_Iv{%rARr(hARr=id2nSS'
        'Iv^rxZggd5WpW}c3LqdLARr(jZE0?4ZFOxTIv_C$ARr(heJlzfARr(ja&>NBVRUq5ZE$oVIv{%rARr(hARr=id2nSSIv^rxZ'
        'ggd5WpW}c3LqdLARr(jV{dMAbRs$+F$y3cARv7#3LqdLAR=gGVPs!(Xkj8cAbScRARr(hAR=^maAhJoAR=>ga%pa7A}k6ZAR'
        'r(hAR=&KbaZ8MZX!A$B3@fCEjeK=W?g$UFnuH<3LqdLAbl(fARr(hB64MMb!BsOUuk3_Iv{%rARr(hARr=id2nSSIv^r*baH'
        '8KXCf>LARr(hARr=eVRUq5a&96zAR=B{K`mNgEqX95IbB;pEm~nMdN3_HE?+HOdoV0EGJRV?Em~nMdN3_HT_hq3ARr(heJlz'
        'fARr(jaBN|2Uvp?-GBq|LIv{%rARr(hARr=id2nSSIv^r*baH8KXCf>LARr(hARr=eVRUq5a&96zAR=B{FfBP@EoNPNHZ*-'
        'CA_^cNARv7#3LqdLAR=XMc4=~NZf#|5bYE#?B03;@3LqdLARr(jba`-PB03-'
        '<X>N37XJv9CED9hXARr(hB4clEb95p)ATl&IG&nIhGc+?f3LqdLAbl(fARr(hB4uuNX>xCFZDnqBUv6P-'
        'Wg<EtdkP>RARr(hB4clEb95p)AR=sOW@Te}V{Bz23LqdLAbl(fARr(hB5!wYWpZD2b7gX0X=EZgAbScRARr(hAR=^maAhJoA'
        'R=jQbY*8{aw04WARr(hARr=RZ*FsRB03;3FfuhWH83+d3LqdLAbl(fARr(hB5rSAVPtJ-ZeL<~aA9+EUv+b3WFk5sdkP>RAR'
        'r(hB6N9hWg<EtB4TfEY-'
        'M3?A}k6ZARr(hAR=RLZgX@ZIv{j%b!7@5ARr)p3LqeTED9hXB64MMb!l>CWFk5sTM8f`ARr=hV`yb<VPAG-'
        'a&u{KZXzrSARr(hB64MLZ*ysMZ*qAeED9hXARr=gb#7m2WFjmIARr(hB64+ZUtx4~Wo>YDA}k6ZARr(jXk}q!Uvp?-'
        'A}k6ZARr(ja%FLKWpi|2X=EZS3LqdLAR=&VVQyb@XkjunHX<wvARr(hB4uuNX>xCFZDnqBUuk3_ED9hXARr=TZgy#MZ*Fa6Z'
        'ggL6VQpn1ED9hXARr=dcWz~JUv+b3a$jj=A}k6ZARr(jZf{><WNm3~Ut)Q1VRLg|b#rB8A_^cNT`US9AR=L8WNCD1Z*F01P;'
        'zf@WpZ?BWpg4rAZB4~b7cyB3I'
    ),
}
_CONTRACT_SHA256 = {
    "request_schema": "a951c7fbfd627be947a7fdcb31d215b4e0f63eb0b42b9a8d9ccd281e6ec4997f",
    "driver_return_schema": "be01d6aba199b67ae44f793c26d8e5ca12de3a88623c9fd837f75f78de7a6f45",
    "operation_table": "ab5b034f58bfc5aaac34bb2e8cd48555782d85c0d4b59ce6b086cafdbcdc810e",
    "approval_marker_schema": "66ce6eadd5cb1e16e24c1c13c8def3fa70d15ceef160e3da305c6f2a6bad7dc7",
    "status_aware_vectors": "d70a6e65c13fce245d3bd60a806502cbe2c83c76926cec8dcd359d8406f529f8",
}

# Phase B R2 closed source contracts. The values are zlib-compressed base85
# encodings; hashes below always bind the original reviewed JSON bytes.
_PHASE_B_R2_CONTRACTS_B85 = {
    "request_schema_v3": (
        'c-rk;S##Sq5PsLM@M!vI#v~<@Qr_Go)9Ep-C-F;@aXBEll#uW$2>OUe|Ggk}@c=+bmh8BrCvgCBcL9F8`(Xj<ad37vqz@Sk-<}Oq'
        '?<@FpcwV+;B+Y<O3_0t0WG$>M_va8J_ZCQw(`3`mdi*rWDQg&24U6;T{RI9kAg3-4>i4d-<&@xLHauUqXNXcDrP$#=UoN%JNc*m9'
        ')0^$(8&|t|)X!~mQS1I`*ieb%EP@ykiYPPD&F5%m)l4eV!kfpjzGw-*OUbt|O7C1RO1%TihWm`oJp}G;93NP;kD`bP_R|0=%X6OW'
        '+PkmXW1L*oAz;{>E))c~oLv-MbwMICxZh-Ebi?$lAX%CN4{F01qM#v-qy_tBa81MJu(JpyDkIGVgD9WpV7+04v1we?21$S!jcH0A'
        'M9&R>D<X`v7=Y&5r)rk8>p8F|F5dlpd5=rA@aAhY<jVI^65xOhzXcTe+^fvKNDn0dM^q$ptiumV><u!U^mv7uXY>t5Z8&Q$X%8*L'
        '_p~5vSSSY}@<PK5J#PZOWy08ZEJL^KacJu%GOW<F4QCXB<Crj%st7}&0%QlF<2w^D9e|19IEL@*wx>Ij5wi3N1U`abWa@GS8%VJE'
        'm<x4{OcYM0=G2~A77Fai*fJb&kmdMb1rszL8KH%yj=|TJ11&NaC$}tLFxM-AXIl1?v1gA(*n{L2lQb#wlgk2rO6c)*=D^TSzz3Ys'
        '{P^lh*e`Q0MP3BQSLidayD15Ph*0vS>0no;hO$ia39!i4_P$kTkp-O2G3vB<TEbkR+ZsRw907i(CD1&A5`QkRXWK6@Y11#z7-$sD'
        '#KOx?q2#htAhz2Qij;&}oZfaP+R~u(JB>wPAl&|-$Rg~QSw4is?+TvPBjODojduwU7x~P+n=G`M#dXsfS^8+omBH{JdDa>^hGmVp'
        'Phxfp%Qj1Qkf?V|XEL=%x<Cjm!$tRTnGdl%^#>%gDst91%Ah5_V(6Ga(D0Hb^WpOvAsZ3X2Wh^SCTLNL<L9N`WYV7jltOl2jO`DK'
        '{3s@dZx@sjuyG-@YP;#<@*#?H=&<Q-FyC!yvJK*EsjC69d-LUr?J0!4CT3sW&RMVS>uX1H;q#W8K0|(ep2Y@>E&lxAa}=`3^Br7L'
        'M8JGq5`;~AWY`ni=1VD}$PFOG30{{Kj!YE8kU^d;11rm!*|{3IDMvBZC-!-_)ij-MYh%R6OrjBDepL&DK-jxO(K=`ze@2tk&43{}'
        'nXxeI7Rk4S4a>bFn1XO_DJg*edhY$G$jjHWoSRBaDV5@CNNM6@w!Wv52qczW`2mY#q^t%Z`h01h7eD#YksI3tkx7LqY<yqpx&00)'
        '_aLRjst8n7WbFwW86Nh};hk)I_=H|(LatXNK?F}p>6N%2vIcjgoL=I91j3pWA_P5jWd2GaE*}{+JJ3D91Fu>IWhqYGOZ<k`W~UKw'
        '+AIEC#GN#?k%<}RGZeLyZP()5rD<zn+3CA!wJh3?55J}bVeZ1WSUL~=b5U0yfUXq861(NVJ_yA7w2NS`THfq&h2Fu7aMp)41-vO>'
        'PL4Ys_GIrde3?09nPdi(!{&F^uT-j5GKRCi@DkRn2?2vRrVsrF2}6Whm{_JbmWm769W{NW_stq50v;xCT)Bav>$3aM{pvnzIu+cu'
        '5c<Fj(|O^!!Ugj_FHi~6a<eLKIS14&2EUL&UK%{Oo<077tl_S-_BKdi71pwDu!GI~g$mz+cEG9<e)`|3(meCCgkZS6OkO33Ez9t3'
        '40N6b3LN3;uTR&*U8Qq&<#@6$Dz<W=NJB2zt71=t`5xvxL@5fMCt}Mo_p|!5EQKG!k44GzIDTACt8}O@=1K(ej3F9Mgy2L7PK4k@'
        '2woW>U_mVKkpE4c68!RE2S%_FFXseDEeotoI7RAHq&`LJQ>1>Sk$SuDB;5ada#@Vg`h~W<Ft-=i>x;LYi_be0(6$5c)AA(#D;&aR'
        '$FBq@C?v~wA3t1Pe|Ya+{`uZzm+#%5FW>#HuCKOz$lZ<R%$Ap?5uijgUED2ZnHzHq>vP}E4oBZ!MbB=6xY{o)p7#~Ed+W_AC18aL'
        'OK+J(rEsCN{8VCh3LH(5L#c<TU1EWX(<wU2Kp{Bq5}1V>g|f`sAH&o$ZPx%lLO4F&K4m5Tf4EDT=LNV)5qlCgxgC{ho%TOZ`=6)%'
        'PnB@@RD(nd3<004PAnBU5G9wBFR~>^6;Gt_Af&LFPWGdO_0auVNMp-y6rlX7SkUIQWJTtxcl=_6Xlcv(gdt;=u_PZW1crLmZ6NcB'
        'mMJn{;A3!e>a34t3P|7%uSI><1`^<?f@ZPvN#n+`&fX7<xJQiF2POA74LevF*fp;G4+q`<F%bXFfbc&8L?8GIzs_%c`(FQ?o%1n{'
        '@JpJ$L%)&}yMxt)$|rL%_z#Su^l|'
    ),
    "seed_bundle_schema": (
        'c-rk+$!^;)5WVXw0-?vKcAPkXb90lR=z=zAd})JJprx_RR1#H^N@4{5_fqmES=`(>8=#9J52=~q@Xf-Z9-O^Bmrql0Pxf3gyn%qb'
        '`#CM8Bn8SbFfsBVD1ycSLjM*h!z5Wz1w9}M<AQKd5N7-H*%VU3Feh{>o9lkgB8<EkfHUusQ=E*w!34zY*6GjwJ<i3z`<uwE@Wcs+'
        'g!?!wvs|jr3NG&ck{;vaD<oq+k*F3B8ZsFZL@An9czogz=Ti^YM08&#;ss^NQEu18h{U>!5lSN1pb07@SuChm6>7l;Cjm}TRARmu'
        'y6AXW8X^w$p`(!(lUr~r^y|{US@{+BhGaAmB5GPG*QZ}&nW+R$rAbv_DfCl=O~PqA*y$w7peV7;1}IB2>hRPqP=*T~HVpVMU__0R'
        'O{}Q)WAr?B;fRib`)q>#S6+%Z2bv?${e9K;ddM4HKOD7Z?_68qoC3UnbQB1@3jhijVm{R?Aazl*o6f^NM?6!FmSp4}F$OBd(tKZW'
        'TtM}p-B!&(_lr5*UQaS*CdKI(^GP=JDg4cVDP7WTLyQnqtC2gBKLHd98@z}DLlRE?;qlQ6N*8|E=^j5;0jX0QAJv_r(8jUQ&&|Ps'
        'DTJy>xd6E(zKr{?-yfs>>JeA6Bov02i`<E#i?R=oP{hC(JOT>2x-swx2@N^(;``;mwdE-GZerlqDzbIUG8MOk`NbO}wuK7Qh@kN4'
        'BBCn0J-Lsm(%fTxOO|kU=UchCl*@TGs8e96Kp-@UvPqOpqHGeSW%LliBOIdOk6o(#cy5s?S1i@Ja_7g#rN|}<Hc_yNf=v|MG79D|'
        'p!{uLtekQPeZ+Uw@(|_c%$q-S)(?Z_!%6Momql}At#Cl%G=evvvXCHL2}QYyIT#Yn>5udN;QY+*e>?NV<C*`VfBMBNBO!hYNhp@-'
        '-%-pJQG8{QAt4c<L}rquCY)w9_&6FRbgotF3hVm3nvSfzdn#>G>KcmLexIowXE<?dkI$xqt?`IstYQo`iADWj7b0x^(&P*lKz1|%'
        '&^Gu<FFsURN*_1uvQkxG5Fs4z_MW9+`G#*<7|Xy<vUsSdlGfUHy3L(Ub7%9)?`#Yq<uMA#3SLQFbe912GX~V?2I>Ra+UeSz?qRom'
        '*lUU#b8%A*#oLuO<uLf#3ZD&38zA>8GyPmeQNb03NwH7Qmh|VcDFlF1o>xV&B1ah<v)nOpWOyx!RpGRbm(|xFy2?mW#8{fw&JvDf'
        'vu!qdzV`gLSbKfd)lZu0A1>8CJKFew$XTeQS7zq?1C6XY3I'
    ),
    "owner_seed_claim_schema": (
        'c-pO0O;5r=5WVv&$i`!<$X9|l1TJa}Q7(qyW}!n_((cyXDM$$a-D&wqp=)B)OVhrcynS!xO}CY5mEk4<c2{M3x`Y`rFT)a}6EG;j'
        '(#e((!e)_>F<*hPQmYf39cX2dfZH6{dHZq$iMA58*|=)iLmElp#{g*kS2R+E{%8SIebamEe#-0`xL!u=j8j^h0#|7~1>;muZ=KgG'
        '1ch7%#p0u1o8QX3BOXij1S-Tu*@}*!$N;2(zQ>2}ykAnyWFn#y9f8BTpC*AoD1yzoAL|vc12Vbj+5B%xQJ4^r+lBZEPa+V(WV2-N'
        '<C@<Q{(Q1+)^=CSMT!RCA6b_=HA+O8)_SY`YvOkEl31@57@k^sxCfu5BBTI$rTEn9twyWX=$zXJLk-WAVm3=xn0qgS?r6~G-RC~1'
        '*FL}R_8!kwDa{j50U6?JBeAep1!YrBNfWB93{47qy2^o)vhtBi1HmbX7YkZmL)Q|O7Nr>ZLMqGoa)iS@S?5^Xz5TXVgimtUA_yds'
        'JrNCyzo@y0EL3)tFE{x~NB'
    ),
    "seed_action_journal_schema": (
        'c-rk(O>f&U487-91jdf>k``%!_Le2ZfNj`{HJ7dE1leX{WwzwWbl0HBe=qIYjz8p}9~lM|xyh9DM3Ilr;(f2*w{RVSb>6p>vjP{b'
        'L28S%ia?`)m}c(DLbjDkwB!N11sw||%L$rtm5h-LN10(Wc-){NQn5g#*QT}Brz8;E4goN>uTTrQwC6t1>f!Xq<W8i=!24BT)i8xb'
        'kjR8rg)D9mXDHU7u!ll^0l7rK?V1XpiU2YR>H9kW&LSYBD<U40aAdqZwv))?2vy^FVTbA#tlV<bh{57@o+8&vKo<M_DL4^F1fAx_'
        '`gJ|B&$zv~xgU=<?<^B2YJfeWa+h*Vsi6c3T$_L^6w8C$qFH8-kz-P~RZvn6EJ0Ut@EAc6W=T96jp{S}E}uC$J1ZwPl?*Ng`ia9d'
        '{7E2gkZV_9I#Sn!PYJ_(;2!zBS71kZYv!De{{brT?fB&MIlxAeoQ?~UWT*5Rr&(#pg$A!)dwKeIW~d{+rs#Cst*F=18SZ%kN!fUC'
        'S>@$|2Qk!$?rKPn3;})R8MW>5+tu7M?V%|7#vQ{<Er(AO#UTZRb}ZtKL)L-f-fJqbu3LLxRb9Yd#-^?0Jcmu`RT_@GI`tajh%yHU'
        'kvoXo8x=Vsy615B_b!?Hc+pU~a#K_1UhqUKpboWmsI@m<Yw09US2FTcBY2GCOCIL*i`>|cgX8UBUJTA#gP)Bj{MHdyg;4<i(G@}?'
        'YEi$LCJ&7AGyQ%snP1G<<m-&laK=7Nrl0F5Qn3i6N3OEJT3}5V%DXs<+CeV2mnlv_o#ct?<&bTHwff?A_uJ}St6gZNQEaz-hJMZ5'
        'Y#CpshNq|bq1f@b>G%^EwZ2oecTs#d>&!&xOl&VNz1?NE={*3SvRMc'
    ),
    "seed_failure_receipt_schema": (
        'c-pN~O;5r=5WV+Tkc~%3)F_GG_^3$(X|aMH2yT{k3dLo&?gwfJ|D7!sZ0+ia<Gy+G@@D4k)@d}5`pN*^HxQq%AX2EAW0~L?2$g~~'
        'm90SxCZi>+3EqH^DQ73#R3gr#GQiaV*fk5jB1+3L513w?_q2E$q&p0H1GkSS{=g4jeAKL${9ufq-OzjRI^CgLW;~695jI_Jh&#Qp'
        '7x*}IA17Yu_EABaRSXFrvTFP|x>O=%i<355r)&spp_a9w1gBt8D2?<>%OFfsYIn5aS};PfT$1bC_W5+*82?~cK$0t;PnORmhz(KZ'
        '{PJ(lwQXiZDG>SM(d+cGbw^ssY}>xsU7(6OgHMGcB5nIEnL5%n6CgR=Ky2gBxj7VJKPHYDRYJqJj#FrvT)gX49CW7Es8<7BO;b4+'
        '_N!WN^3O(V>!=TGfZ4%1iQ_bHCUjK(u?e9PoSm}=1ws3f'
    ),
    "capability_4d_schema": (
        'c-rk&O>dh(5WV|XSgRh5WVhhTIoF3uRnuCHdWh2100V4|m+kHlH?sWSyI??IgOjGtrOI(<*f(EqmbV*?$BZ5nva2!6{GS*Qn-n&s'
        'GliN28td6vi?wM4lgZ|6EkPy?6<4pne_pQm>iXu}lCPFuZ$7S;A6WU-esZ`R*xMvB$%7P(X@j85H}-s6oy$N8f;Nykz6Kd2Y|`f='
        'h*@R7>x9x6wejon3~3?b28(Tk{Dd;5#5!jk00JE&ag)GPa1;=sF8acL+)kYZIMLnCo$lYTK2Xw#7ew7=dL5<NmWLvWij3cS^Jl})'
        '-D7x5gL@I8<{FK%!|qMJ#hd!lF?UM4A)vf#YoQ297^7~@`Y<E0Dd{nrPP-oXU3t94qH*k7Oo;_0xo<Q7m7^gp!lB4Vy%`|KF?<Mw'
        '+;Y)&=d!?l7Cvx)?!Ex(2|peMeuO7wl+jT%R<vAb47*!hjsNU~2Kw9Aoq^iw7okZ8!cY_s52Rn(yo?|<*x~wGW?RX@Hz>*G1fdVX'
        '`jzjmeE(1R4oY=N()`Z7ckgGO=XSo9He(EVkl?>9#ahS{Vm30u4V<Et_mJ8hbZW`qD2<1>y^|JF<@7?MZiOjzi-<%}<}_HsuoCLo'
        'SFUoC@F2@2{nay3N1OHngA#m`Qzf@90schBm$Mz3%y27*dvd<0XHb7s6>(ZXt#13}YD-z3YQ!FacTSxRN~^Q~q3+W5E#;_c#rZbu'
        'M}GiPt_3{'
    ),
}
_PHASE_B_R2_CONTRACT_SHA256 = {
    "request_schema_v3": "2d6b66eab9414dfe7d05b36f2c11ff862f84d4fcb1e840ac543922e16a5a5d43",
    "seed_bundle_schema": "58e458239ed6a928f875be991c42e49586b7d705f6c91608c5936043cb2fe3a4",
    "owner_seed_claim_schema": "d0b12d73d9f0bb8c63907d232a01c5d2103bdb4665cddbe915a12c627a097d5a",
    "seed_action_journal_schema": "442250c9943bb64dddebf2381835df35a6f806568675f132cb899759c274852e",
    "seed_failure_receipt_schema": "f2bd5225e646d5aca4808600bdadd6265307e6086df200d84b065b1e3be9beee",
    "capability_4d_schema": "eadc359e21b31e473aeedf4ac9787d784943df2eb1f73d964ffb215b41e51828",
}


class RehearsalError(RuntimeError):
    """Closed, privacy-safe validation failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _contract_bytes(name: str) -> bytes:
    try:
        if name in _PHASE_B_R2_CONTRACTS_B85:
            raw = zlib.decompress(base64.b85decode(
                _PHASE_B_R2_CONTRACTS_B85[name].encode("ascii")
            ))
            digest = _PHASE_B_R2_CONTRACT_SHA256[name]
        else:
            raw = base64.b85decode(_EMBEDDED_CONTRACTS_B85[name].encode("ascii"))
            digest = _CONTRACT_SHA256[name]
    except (KeyError, ValueError, UnicodeError):
        raise RehearsalError("EMBEDDED_CONTRACT_UNAVAILABLE") from None
    if sha256_hex(raw) != digest:
        raise RehearsalError("EMBEDDED_CONTRACT_DIGEST_INVALID")
    return raw


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _parse_float(_value: str) -> Any:
    raise ValueError("floating point is not permitted")


def _parse_constant(_value: str) -> Any:
    raise ValueError("non-finite number is not permitted")


def _strict_loads(raw: bytes | str) -> Any:
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(
            text,
            object_pairs_hook=_request_object_pairs,
            parse_float=_parse_float,
            parse_constant=_parse_constant,
        )
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        raise RehearsalError("JSON_INVALID") from None


def _canonical(value: Any) -> bytes:
    def visit(item: Any) -> None:
        if item is None or type(item) in {bool, int}:
            return
        if isinstance(item, str):
            if any(ord(char) < 0x20 or ord(char) > 0x7E for char in item):
                raise ValueError("non-printable canonical string")
            return
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str) or any(
                    ord(char) < 0x20 or ord(char) > 0x7E for char in key
                ):
                    raise ValueError("invalid canonical key")
                visit(child)
            return
        raise ValueError("unsupported canonical type")

    try:
        visit(value)
        return json.dumps(
            value, ensure_ascii=True, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError):
        raise RehearsalError("CANONICAL_VALUE_INVALID") from None


def _is_type(value: Any, wanted: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "boolean": type(value) is bool,
        "null": value is None,
        "number": type(value) in {int, float} and type(value) is not bool,
    }.get(wanted, False)


def _schema_matches(value: Any, schema: Mapping[str, Any]) -> bool:
    if not isinstance(schema, Mapping):
        return False
    if "type" in schema:
        kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(value, kind) for kind in kinds):
            return False
    if "const" in schema and not _strict_equal(value, schema["const"]):
        return False
    if "enum" in schema and not any(_strict_equal(value, item) for item in schema["enum"]):
        return False
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            return False
        if len(value) > schema.get("maxLength", len(value)):
            return False
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            return False
    if type(value) in {int, float} and type(value) is not bool:
        if value < schema.get("minimum", value):
            return False
        if value > schema.get("maximum", value):
            return False
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            return False
        if schema.get("uniqueItems") is True:
            for index, item in enumerate(value):
                if any(_strict_equal(item, previous) for previous in value[:index]):
                    return False
        if "items" in schema and any(
            not _schema_matches(item, schema["items"]) for item in value
        ):
            return False
    if isinstance(value, dict):
        if len(value) < schema.get("minProperties", 0) or len(value) > schema.get("maxProperties", len(value)):
            return False
        if any(key not in value for key in schema.get("required", [])):
            return False
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, child in value.items():
            if key in properties:
                if not _schema_matches(child, properties[key]):
                    return False
            elif additional is False:
                return False
            elif isinstance(additional, dict) and not _schema_matches(child, additional):
                return False
    if "oneOf" in schema:
        matches = sum(_schema_matches(value, item) for item in schema["oneOf"])
        if matches != 1:
            return False
    if "anyOf" in schema and not any(
        _schema_matches(value, item) for item in schema["anyOf"]
    ):
        return False
    if "allOf" in schema:
        for clause in schema["allOf"]:
            condition = clause.get("if")
            matches = isinstance(condition, dict) and _schema_matches(value, condition)
            branch = clause.get("then") if matches else clause.get("else")
            if branch is not None and not _schema_matches(value, branch):
                return False
            if not any(key in clause for key in ("if", "then", "else")) and not _schema_matches(value, clause):
                return False
    return True


def _load_contract(name: str) -> Any:
    raw = _contract_bytes(name)
    value = _strict_loads(raw)
    if _canonical(value) != raw:
        # Source design files are formatted JSON, not canonical wire artifacts;
        # validate their pin separately and canonicalize only their instances.
        return value
    return value


LEGACY_REQUEST_SCHEMA = _load_contract("request_schema")
REQUEST_SCHEMA_V3 = _load_contract("request_schema_v3")
REQUEST_SCHEMA = REQUEST_SCHEMA_V3
SEED_BUNDLE_SCHEMA = _load_contract("seed_bundle_schema")
OWNER_SEED_CLAIM_SCHEMA = _load_contract("owner_seed_claim_schema")
SEED_ACTION_JOURNAL_SCHEMA = _load_contract("seed_action_journal_schema")
SEED_FAILURE_RECEIPT_SCHEMA = _load_contract("seed_failure_receipt_schema")
CAPABILITY_4D_SCHEMA = _load_contract("capability_4d_schema")
DRIVER_RETURN_SCHEMA = _load_contract("driver_return_schema")
OPERATION_TABLE = _load_contract("operation_table")
APPROVAL_MARKER_SCHEMA = _load_contract("approval_marker_schema")
STATUS_AWARE_VECTORS = _load_contract("status_aware_vectors")
CANONICAL_OPERATION_TABLE_SHA256 = sha256_hex(_canonical(OPERATION_TABLE))
CASE_ORDER = tuple(row["case"] for row in OPERATION_TABLE)
_CASE_INDEX = {case: index for index, case in enumerate(CASE_ORDER)}
_CASE_METHODS = {
    "P2-app-create": "probe_trigger_create_server",
    "P2-app-update": "probe_trigger_update_server",
    "4a": "cas_principal_ff",
    "4b-nff": "probe_principal_nff_server",
    "4b-delete": "probe_principal_delete_server",
    "4c": "classify_owner_route_no_push",
    "4d-branch-create": "create_once_fixture",
    "4d-branch-update": "probe_immutable_update",
    "4d-branch-delete": "probe_immutable_delete",
    "4d-tag-create": "create_once_fixture",
    "4d-tag-update": "probe_immutable_update",
    "4d-tag-delete": "probe_immutable_delete",
    "4e": "probe_canonical_tag_create",
    "4f-create": "probe_restricted_create",
    "4f-move": "probe_restricted_move",
    "P5-event-observe": "read_existing_4a_app_event",
    "P5-duplicate-fence": "reject_existing_same_request_fence",
    "4g": "return_owner_action",
}


@dataclass(frozen=True, slots=True)
class ValidatedRouteRequest:
    request: Mapping[str, Any]
    request_bytes: bytes = field(repr=False)
    request_sha256: str
    plan_sha256: str
    owner_attestation_sha256: str
    request_id: str
    request_path: str
    commit_sha: str
    parent_sha: str
    workflow_blob_sha: str
    expected_vmm_sha: str
    called_code_blobs: Mapping[str, str]

    @property
    def plan(self) -> Mapping[str, Any]:
        return self.request["plan"]

    @property
    def owner_attestation(self) -> Mapping[str, Any]:
        return self.request["owner_attestation"]


MAX_ROUTE_REQUEST_BYTES = 12 * 1024 * 1024
MAX_SEED_JOURNAL_BYTES = 4 * 1024 * 1024
MAX_SEED_RESPONSE_BYTES = 1024 * 1024
_SEED_ROLES = ("principal", "P2-update", "4f-move")
_SEED_REFS = {
    "principal": "refs/heads/rehearsal/principal/cyax125-20261101-route",
    "P2-update": "refs/heads/rehearsal/trigger/cyax125-p2-app-update",
    "4f-move": "refs/heads/rehearsal/restricted/cyax125-20261101-move",
}
_SEED_TARGET_SHA = "f12fbb8ec52e4c7510564f602a15f26173fe772e"
_SEED_TARGET_TREE = "c515bf2745061da9150285d6d7c6035b061a1caa"
_SEED_COMMENT_PREFIX = "CYAX125_SEED_V1:"
_ISSUE_125_URL = f"https://api.github.com/repos/{REPOSITORY}/issues/125"


def validate_route_request_bytes(raw: bytes) -> ValidatedRouteRequest:
    """Validate the closed schema-3 route request and its acyclic digests."""

    if not isinstance(raw, bytes) or not raw or raw.endswith(b"\n") or len(raw) > MAX_ROUTE_REQUEST_BYTES:
        raise RehearsalError("ROUTE_REQUEST_BYTES_INVALID")
    request = _strict_loads(raw)
    if not isinstance(request, dict) or not _schema_matches(request, REQUEST_SCHEMA_V3):
        raise RehearsalError("ROUTE_REQUEST_SCHEMA_INVALID")
    if _canonical(request) != raw:
        raise RehearsalError("ROUTE_REQUEST_NOT_CANONICAL")
    plan = request["plan"]
    attestation = request["owner_attestation"]
    plan_bytes = _canonical(plan)
    attestation_bytes = _canonical(attestation)
    plan_sha = sha256_hex(plan_bytes)
    attestation_sha = sha256_hex(attestation_bytes)
    if plan_sha != request["plan_sha256"] or attestation_sha != request[
        "owner_attestation_sha256"
    ]:
        raise RehearsalError("ROUTE_REQUEST_DIGEST_MISMATCH")
    if (
        plan["operation_table_sha256"] != CANONICAL_OPERATION_TABLE_SHA256
        or tuple(plan["case_order"]) != CASE_ORDER
        or plan["called_code_blobs"] != request["called_code_blobs"]
        or plan["workflow_blob_sha"] != request["workflow_blob_sha"]
        or plan["candidate_sha"] != request["implementation_sha"]
        or request["expected_vmm_sha"] != plan["fixture_base"]
        or request["request_id"] != attestation["request_id"]
        or plan["intent_id"] != attestation["intent_id"]
        or plan_sha != attestation["plan_sha256"]
        or plan["candidate_sha"] != attestation["candidate_sha"]
        or plan["repository"] != REPOSITORY
        or plan["owner_actor_id"] != OWNER_ID
        or plan["app_id"] != APP_ID
        or plan["installation_id"] != INSTALLATION_ID
        or request["seed_receipt_bundle"]["intent_id"] != plan["intent_id"]
        or plan["seed_receipts_sha256"] != sha256_hex(_canonical(request["seed_receipt_bundle"]))
    ):
        raise RehearsalError("ROUTE_REQUEST_BINDING_INVALID")
    return ValidatedRouteRequest(
        request=request,
        request_bytes=raw,
        request_sha256=sha256_hex(raw),
        plan_sha256=plan_sha,
        owner_attestation_sha256=attestation_sha,
        request_id=request["request_id"],
        request_path=f"lifecycle-requests/{request['request_id']}.json",
        commit_sha="",
        parent_sha=request["implementation_sha"],
        workflow_blob_sha=request["workflow_blob_sha"],
        expected_vmm_sha=request["expected_vmm_sha"],
        called_code_blobs=dict(request["called_code_blobs"]),
    )


class SeedReceiptError(RehearsalError):
    """Fail-closed seed validation error with a closed, hash-only receipt."""

    def __init__(self, receipt: Mapping[str, Any], observations: Any = None) -> None:
        super().__init__(str(receipt["reason"]))
        self.failure_receipt = dict(receipt)
        self.observation_preimage = tuple(dict(item) for item in (observations or ()))


@dataclass(frozen=True, slots=True)
class SeedReceiptEvidence:
    status: str
    bundle_sha256: str
    verified_roles: tuple[str, ...]
    live_observation_sha256: str
    request_id: str
    request_sha256: str
    plan_sha256: str
    observations: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "bundle_sha256": self.bundle_sha256,
            "verified_roles": list(self.verified_roles),
            "live_observation_sha256": self.live_observation_sha256,
            "request_id": self.request_id,
            "request_sha256": self.request_sha256,
            "plan_sha256": self.plan_sha256,
            "observations": [dict(item) for item in self.observations],
        }


_SEED_OBSERVATION_FIELDS = frozenset({
    "method", "path", "role", "kind", "started_at", "completed_at",
    "status", "headers", "body_b64", "body_sha256", "body_bytes",
})


class _RecordedSeedTransport:
    """Replay retained seed observations through the live producer validator."""

    def __init__(self, entries: list[tuple[str, ApiResponse]]) -> None:
        self._entries = tuple(entries)
        self._index = 0

    def get(self, path: str) -> ApiResponse:
        if self._index >= len(self._entries):
            raise ValueError("SEED_REPLAY_OBSERVATION_MISSING")
        expected_path, response = self._entries[self._index]
        if path != expected_path:
            raise ValueError("SEED_REPLAY_PATH_MISMATCH")
        self._index += 1
        return response

    def require_complete(self) -> None:
        if self._index != len(self._entries):
            raise ValueError("SEED_REPLAY_OBSERVATION_UNUSED")


def _seed_failure(
    status: str,
    reason: str,
    *,
    role: str | None = None,
    journal_sha256: str | None = None,
    known_ref_sha: str | None = None,
    observations: Any = None,
) -> SeedReceiptError:
    preserved = [dict(item) for item in (observations or ())]
    observation_bytes = _canonical(preserved)
    receipt = {
        "schema_version": 1,
        "status": status,
        "reason": reason,
        "role": role,
        "journal_sha256": journal_sha256,
        "known_ref_sha": known_ref_sha,
        "observations_sha256": sha256_hex(observation_bytes),
        "next_case": None,
        "unresolved": True,
    }
    if not _schema_matches(receipt, SEED_FAILURE_RECEIPT_SCHEMA):
        raise RehearsalError("SEED_FAILURE_RECEIPT_SCHEMA_INVALID")
    return SeedReceiptError(receipt, preserved)


_SEED_OBSERVATION_HEADERS = frozenset({
    "content-type", "date", "etag", "last-modified", "x-github-request-id",
})


def _seed_live_observation(
    *, role: str | None, kind: str, path: str, response: ApiResponse,
    started_at: str, completed_at: str,
) -> dict[str, Any]:
    body = bytes(response.body)
    if len(body) > MAX_PROVIDER_BODY:
        raise ValueError("SEED_LIVE_RESPONSE_OVERSIZE")
    headers = {
        str(key).lower(): str(value)
        for key, value in response.headers.items()
        if str(key).lower() in _SEED_OBSERVATION_HEADERS
    }
    return {
        "method": "GET", "path": path, "role": role, "kind": kind,
        "started_at": started_at, "completed_at": completed_at,
        "status": response.status, "headers": headers,
        "body_b64": base64.b64encode(body).decode("ascii"),
        "body_sha256": sha256_hex(body), "body_bytes": len(body),
    }


def _seed_get_observation(
    transport: "AnonymousTransport", observations: list[dict[str, Any]],
    *, role: str | None, kind: str, path: str,
) -> ApiResponse:
    started_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    try:
        response = transport.get(path)
    except Exception:
        observations.append({
            "method": "GET", "path": path, "role": role, "kind": kind,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "outcome": "TRANSPORT_EXCEPTION",
        })
        raise
    observations.append(_seed_live_observation(
        role=role, kind=kind, path=path, response=response,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    ))
    return response


def serialize_seed_receipt_evidence(evidence: SeedReceiptEvidence) -> bytes:
    if not isinstance(evidence, SeedReceiptEvidence) or evidence.status != "PASS_ACCEPTED":
        raise RehearsalError("SEED_EVIDENCE_INVALID")
    value = evidence.to_dict()
    value["evidence_sha256"] = sha256_hex(_canonical(value))
    return _canonical(value)


def _parse_seed_observation_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("SEED_OBSERVATION_TIME_INVALID")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("SEED_OBSERVATION_TIME_INVALID") from None
    if (
        parsed.tzinfo != timezone.utc
        or parsed.isoformat(timespec="microseconds") != value
    ):
        raise ValueError("SEED_OBSERVATION_TIME_INVALID")
    return parsed


def rehydrate_seed_receipt_evidence(
    raw: bytes, request: ValidatedRouteRequest,
) -> SeedReceiptEvidence:
    try:
        value = _strict_loads(raw)
        if _canonical(value) != raw:
            raise ValueError
        if not isinstance(value, dict) or set(value) != {
            "schema_version", "status", "bundle_sha256", "verified_roles",
            "live_observation_sha256", "request_id", "request_sha256",
            "plan_sha256", "observations", "evidence_sha256",
        }:
            raise ValueError
        digest = value.pop("evidence_sha256")
        if digest != sha256_hex(_canonical(value)):
            raise ValueError
        if (
            type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["status"] != "PASS_ACCEPTED"
            or value["request_id"] != request.request_id
            or value["request_sha256"] != request.request_sha256
            or value["plan_sha256"] != request.plan_sha256
            or value["bundle_sha256"] != sha256_hex(_canonical(request.request["seed_receipt_bundle"]))
            or value["verified_roles"] != list(_SEED_ROLES)
            or not isinstance(value["observations"], list)
            or len(value["observations"]) != 7
        ):
            raise ValueError
        expected_paths = [
            item
            for record in request.request["seed_receipt_bundle"]["records"]
            for item in (
                (record["role"], "comment", f"/repos/{REPOSITORY}/issues/comments/{record['owner_receipt_comment_id']}"),
                (record["role"], "current_ref", f"/repos/{REPOSITORY}/git/ref/{record['journal']['ref'].removeprefix('refs/') }"),
            )
        ] + [(None, "seed_commit", f"/repos/{REPOSITORY}/git/commits/{_SEED_TARGET_SHA}")]
        replay_entries: list[tuple[str, ApiResponse]] = []
        for observation, expected in zip(value["observations"], expected_paths):
            if not isinstance(observation, dict) or set(observation) != _SEED_OBSERVATION_FIELDS:
                raise ValueError
            if (observation.get("role"), observation.get("kind"), observation.get("path")) != expected:
                raise ValueError
            if (
                observation.get("method") != "GET"
                or type(observation.get("status")) is not int
                or observation["status"] != 200
            ):
                raise ValueError
            started_at = _parse_seed_observation_time(observation["started_at"])
            completed_at = _parse_seed_observation_time(observation["completed_at"])
            if started_at > completed_at:
                raise ValueError
            body = base64.b64decode(observation["body_b64"], validate=True)
            headers = observation["headers"]
            if (
                base64.b64encode(body).decode("ascii") != observation["body_b64"]
                or type(observation["body_bytes"]) is not int
                or len(body) != observation["body_bytes"]
                or len(body) > MAX_PROVIDER_BODY
                or sha256_hex(body) != observation["body_sha256"]
                or not isinstance(headers, dict)
                or set(headers) - _SEED_OBSERVATION_HEADERS
                or any(type(key) is not str or type(header) is not str for key, header in headers.items())
            ):
                raise ValueError
            replay_entries.append((
                observation["path"],
                ApiResponse(observation["status"], body, dict(headers)),
            ))
        if sha256_hex(_canonical(value["observations"])) != value["live_observation_sha256"]:
            raise ValueError
        replay = _RecordedSeedTransport(replay_entries)
        replayed = validate_seed_receipt_bundle(request, replay)
        replay.require_complete()
        without_times = lambda rows: [
            {key: item for key, item in row.items() if key not in {"started_at", "completed_at"}}
            for row in rows
        ]
        if not _strict_equal(
            without_times(value["observations"]),
            without_times(list(replayed.observations)),
        ):
            raise ValueError
        return SeedReceiptEvidence(
            value["status"], value["bundle_sha256"], tuple(value["verified_roles"]),
            value["live_observation_sha256"], value["request_id"],
            value["request_sha256"], value["plan_sha256"],
            tuple(dict(item) for item in value["observations"]),
        )
    except Exception:
        raise RehearsalError("SEED_EVIDENCE_REHYDRATION_INVALID") from None


def serialize_seed_receipt_evidence_pair(
    first: SeedReceiptEvidence, second: SeedReceiptEvidence,
) -> bytes:
    value = {"schema_version": 1, "first": first.to_dict(), "second": second.to_dict()}
    return _canonical(value)


def rehydrate_seed_receipt_evidence_pair(
    raw: bytes, request: ValidatedRouteRequest,
) -> dict[str, Any]:
    try:
        pair = _strict_loads(raw)
        if _canonical(pair) != raw:
            raise ValueError
        if (
            not isinstance(pair, dict)
            or set(pair) != {"schema_version", "first", "second"}
            or type(pair["schema_version"]) is not int
            or pair["schema_version"] != 1
        ):
            raise ValueError
        evidence = []
        for key in ("first", "second"):
            encoded = dict(pair[key])
            encoded["evidence_sha256"] = sha256_hex(_canonical(encoded))
            evidence.append(rehydrate_seed_receipt_evidence(_canonical(encoded), request))
        if evidence[0].live_observation_sha256 == evidence[1].live_observation_sha256:
            raise ValueError
        return {"schema_version": 1, "first": evidence[0].to_dict(), "second": evidence[1].to_dict()}
    except Exception:
        raise RehearsalError("SEED_EVIDENCE_PAIR_REHYDRATION_INVALID") from None


def _decode_seed_envelope(value: Any, *, label: str) -> tuple[bytes, Any]:
    if not isinstance(value, dict) or not _schema_matches(value, {
        "type": "object",
        "required": ["status", "github_request_id", "date_header", "body_b64", "body_sha256"],
        "properties": {
            "status": {"type": "integer", "minimum": 100, "maximum": 599},
            "github_request_id": {"type": "string", "minLength": 1},
            "date_header": {"type": "string", "minLength": 1},
            "body_b64": {"type": "string", "minLength": 1},
            "body_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
        "additionalProperties": False,
    }):
        raise ValueError(f"{label}_ENVELOPE_INVALID")
    try:
        body = base64.b64decode(value["body_b64"], validate=True)
    except (ValueError, TypeError):
        raise ValueError(f"{label}_BODY_BASE64_INVALID") from None
    if (
        not body
        or len(body) > MAX_SEED_RESPONSE_BYTES
        or base64.b64encode(body).decode("ascii") != value["body_b64"]
        or sha256_hex(body) != value["body_sha256"]
    ):
        raise ValueError(f"{label}_BODY_DIGEST_INVALID")
    return body, _strict_loads(body)


def _check_seed_response(
    response: Mapping[str, Any], *, status: int, label: str
) -> tuple[bytes, Any]:
    body, value = _decode_seed_envelope(response, label=label)
    if response["status"] != status:
        raise ValueError(f"{label}_STATUS_INVALID")
    if not response["github_request_id"] or not response["date_header"]:
        raise ValueError(f"{label}_REQUEST_METADATA_INVALID")
    return body, value


def _parse_seed_time(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        raise ValueError("SEED_TIME_INVALID") from None


def _seed_claim_body(claim: Mapping[str, Any]) -> bytes:
    if not _schema_matches(claim, OWNER_SEED_CLAIM_SCHEMA):
        raise ValueError("SEED_CLAIM_SCHEMA_INVALID")
    return _SEED_COMMENT_PREFIX.encode("ascii") + _canonical(claim)


def validate_seed_receipt_bundle(
    request: ValidatedRouteRequest,
    transport: "AnonymousTransport",
) -> SeedReceiptEvidence:
    """Validate schema-3 Owner receipts and anonymous fixed current readbacks.

    A valid result records observed procedural evidence. It does not make an
    Owner-authored comment a provider signature or establish provider-verified
    authorship for the historical journal bytes.
    """

    observations: list[dict[str, Any]] = []
    bundle = request.request.get("seed_receipt_bundle")
    role: str | None = None
    journal_hash: str | None = None
    known_ref_sha: str | None = None
    try:
        if (
            not isinstance(bundle, dict)
            or not _schema_matches(bundle, SEED_BUNDLE_SCHEMA)
            or bundle["repository"] != REPOSITORY
            or bundle["intent_id"] != request.plan["intent_id"]
            or sha256_hex(_canonical(bundle)) != request.plan["seed_receipts_sha256"]
        ):
            raise ValueError("SEED_BUNDLE_BINDING_INVALID")
        if len(_canonical(bundle)) > MAX_ROUTE_REQUEST_BYTES:
            raise ValueError("SEED_BUNDLE_OVERSIZE")
        records = bundle["records"]
        if tuple(record.get("role") for record in records) != _SEED_ROLES:
            raise ValueError("SEED_ROLE_ORDER_INVALID")
        if len({record["role"] for record in records}) != len(_SEED_ROLES):
            raise ValueError("SEED_ROLE_DUPLICATE")
        for record in records:
            role = record["role"]
            journal = record["journal"]
            journal_hash = record["journal_sha256"]
            if not _schema_matches(journal, SEED_ACTION_JOURNAL_SCHEMA):
                raise ValueError("SEED_JOURNAL_SCHEMA_INVALID")
            journal_bytes = _canonical(journal)
            if len(journal_bytes) > MAX_SEED_JOURNAL_BYTES or sha256_hex(journal_bytes) != journal_hash:
                raise ValueError("SEED_JOURNAL_DIGEST_INVALID")
            expected_ref = _SEED_REFS[role]
            if (
                journal["repository"] != REPOSITORY
                or journal["intent_id"] != bundle["intent_id"]
                or journal["role"] != role
                or journal["ref"] != expected_ref
                or journal["target_sha"] != _SEED_TARGET_SHA
                or journal["target_tree"] != _SEED_TARGET_TREE
                or journal["create_method"] != "POST"
                or journal["create_path"] != f"/repos/{REPOSITORY}/git/refs"
                or journal["create_payload"] != {"ref": expected_ref, "sha": _SEED_TARGET_SHA}
                or journal["readback_method"] != "GET"
                or journal["readback_path"] != f"/repos/{REPOSITORY}/git/ref/{expected_ref.removeprefix('refs/')}"
                or journal["terminal_state"] != "CREATED_AND_READ_BACK"
                or journal["no_pending_writes"] is not True
                or _parse_seed_time(journal["operation_completed_at"]) < _parse_seed_time(journal["operation_started_at"])
            ):
                raise ValueError("SEED_JOURNAL_BINDING_INVALID")
            identity_body, identity = _check_seed_response(
                journal["actor_identity"], status=200, label="SEED_ACTOR_IDENTITY"
            )
            if not isinstance(identity, dict) or (
                type(identity.get("id")) is not int
                or identity.get("id") != OWNER_ID
                or identity.get("login") != "vmmhep"
                or identity.get("type") != "User"
            ):
                raise ValueError("SEED_OWNER_IDENTITY_INVALID")
            create_body, create_value = _check_seed_response(
                journal["create_response"], status=201, label="SEED_CREATE"
            )
            if not isinstance(create_value, dict) or (
                create_value.get("ref") != expected_ref
                or not isinstance(create_value.get("object"), dict)
                or create_value["object"].get("type") != "commit"
                or create_value["object"].get("sha") != _SEED_TARGET_SHA
            ):
                raise ValueError("SEED_CREATE_PROJECTION_INVALID")
            readback_body, readback_value = _check_seed_response(
                journal["readback_response"], status=200, label="SEED_READBACK"
            )
            if not isinstance(readback_value, dict) or (
                readback_value.get("ref") != expected_ref
                or not isinstance(readback_value.get("object"), dict)
                or readback_value["object"].get("type") != "commit"
                or readback_value["object"].get("sha") != _SEED_TARGET_SHA
            ):
                raise ValueError("SEED_READBACK_PROJECTION_INVALID")
            claim = record["claim"]
            expected_claim = {
                "claim_version": 1, "repository": REPOSITORY, "issue_number": 125,
                "intent_id": bundle["intent_id"], "role": role, "ref": expected_ref,
                "target_sha": _SEED_TARGET_SHA, "target_tree": _SEED_TARGET_TREE,
                "journal_sha256": journal_hash, "owner_user_id": OWNER_ID,
                "terminal_state": "CREATED_AND_READ_BACK", "no_pending_writes": True,
            }
            claim_body = _seed_claim_body(claim)
            if not _strict_equal(claim, expected_claim):
                raise ValueError("SEED_CLAIM_BINDING_INVALID")
            if sha256_hex(claim_body) != record["owner_action_claim_sha256"]:
                raise ValueError("SEED_CLAIM_DIGEST_INVALID")
            comment_id = record["owner_receipt_comment_id"]
            if type(comment_id) is not int or comment_id <= 0:
                raise ValueError("SEED_COMMENT_ID_INVALID")
            comment_path = f"/repos/{REPOSITORY}/issues/comments/{comment_id}"
            try:
                comment_response = _seed_get_observation(
                    transport, observations, role=role, kind="comment", path=comment_path,
                )
            except Exception:
                raise _seed_failure(
                    "BLOCKED_UNKNOWN", "SEED_COMMENT_READ_UNKNOWN", role=role,
                    journal_sha256=journal_hash, observations=observations,
                ) from None
            if comment_response.status != 200:
                raise _seed_failure(
                    "BLOCKED_NOT_VERIFIABLE"
                    if 400 <= comment_response.status < 500 and comment_response.status not in {408, 425, 429}
                    else "BLOCKED_UNKNOWN",
                    "SEED_COMMENT_READ_UNAVAILABLE", role=role, journal_sha256=journal_hash,
                    observations=observations,
                )
            comment = _strict_loads(comment_response.body)
            if not isinstance(comment, dict) or (
                type(comment.get("id")) is not int
                or comment.get("id") != comment_id
                or not isinstance(comment.get("user"), dict)
                or type(comment["user"].get("id")) is not int
                or comment["user"].get("id") != OWNER_ID
                or comment["user"].get("type") != "User"
                or comment.get("issue_url") != _ISSUE_125_URL
                or comment.get("body") != claim_body.decode("ascii")
            ):
                raise ValueError("SEED_COMMENT_BINDING_INVALID")
            current_path = f"/repos/{REPOSITORY}/git/ref/{expected_ref.removeprefix('refs/')}"
            try:
                current_response = _seed_get_observation(
                    transport, observations, role=role, kind="current_ref", path=current_path,
                )
            except Exception:
                raise _seed_failure("BLOCKED_UNKNOWN", "SEED_CURRENT_REF_READ_UNKNOWN",
                                    role=role, journal_sha256=journal_hash, observations=observations) from None
            if current_response.status == 404:
                raise _seed_failure("BLOCKED_NOT_VERIFIABLE", "SEED_CURRENT_REF_ABSENT",
                                    role=role, journal_sha256=journal_hash, observations=observations)
            if current_response.status != 200:
                raise _seed_failure("BLOCKED_UNKNOWN", "SEED_CURRENT_REF_READ_UNKNOWN",
                                    role=role, journal_sha256=journal_hash, observations=observations)
            current = _strict_loads(current_response.body)
            if not isinstance(current, dict) or (
                current.get("ref") != expected_ref
                or not isinstance(current.get("object"), dict)
                or current["object"].get("type") != "commit"
                or current["object"].get("sha") != _SEED_TARGET_SHA
            ):
                raise ValueError("SEED_CURRENT_REF_BINDING_INVALID")
            known_ref_sha = current["object"]["sha"]

        commit_path = f"/repos/{REPOSITORY}/git/commits/{_SEED_TARGET_SHA}"
        try:
            commit_response = _seed_get_observation(
                transport, observations, role=None, kind="seed_commit", path=commit_path,
            )
        except Exception:
            raise _seed_failure("BLOCKED_UNKNOWN", "SEED_TARGET_COMMIT_READ_UNKNOWN",
                                observations=observations) from None
        if commit_response.status != 200:
            raise _seed_failure("BLOCKED_UNKNOWN", "SEED_TARGET_COMMIT_READ_UNKNOWN",
                                role=None, observations=observations)
        commit = _strict_loads(commit_response.body)
        if not isinstance(commit, dict) or (
            commit.get("sha") != _SEED_TARGET_SHA
            or not isinstance(commit.get("tree"), dict)
            or commit["tree"].get("sha") != _SEED_TARGET_TREE
        ):
            raise ValueError("SEED_TARGET_TREE_MISMATCH")
        observation_digest = sha256_hex(_canonical(observations))
        return SeedReceiptEvidence(
            "PASS_ACCEPTED", sha256_hex(_canonical(bundle)), _SEED_ROLES,
            observation_digest, request.request_id, request.request_sha256,
            request.plan_sha256, tuple(dict(item) for item in observations),
        )
    except SeedReceiptError:
        raise
    except (KeyError, TypeError, ValueError, RehearsalError) as error:
        reason = error.reason if isinstance(error, RehearsalError) else str(error)
        raise _seed_failure(
            "BLOCKED_NOT_VERIFIABLE", reason or "SEED_RECEIPT_INVALID",
            role=role, journal_sha256=journal_hash, known_ref_sha=known_ref_sha,
            observations=observations,
        ) from None


def validate_rehearsal_route_trigger(
    repository_path: str,
    *,
    repository: str,
    ref: str,
    commit_sha: str,
    actor: str,
    event: Mapping[str, Any],
) -> ValidatedRouteRequest:
    """Validate the Owner-created one-file trigger and exact package closure."""

    if (
        repository != REPOSITORY
        or not isinstance(ref, str)
        or not ref.startswith("refs/heads/rehearsal/trigger/")
        or not isinstance(commit_sha, str)
        or _FULL_REF_SHA.fullmatch(commit_sha) is None
        or not isinstance(actor, str)
        or not actor
        or not isinstance(event, Mapping)
    ):
        raise RehearsalError("TRIGGER_IDENTITY_INVALID")
    sender = event.get("sender")
    installation = event.get("installation")
    if (
        event.get("ref") != ref
        or event.get("after") != commit_sha
        or event.get("created") is not True
        or event.get("deleted") is not False
        or event.get("forced") is not False
        or event.get("before") != "0" * 40
        or not isinstance(event.get("repository"), Mapping)
        or event["repository"].get("full_name") != repository
        or not isinstance(sender, Mapping)
        or sender.get("type") != "User"
        or sender.get("login") != actor
        or sender.get("login") == APP_LOGIN
        or (isinstance(installation, Mapping) and installation.get("id") == INSTALLATION_ID)
    ):
        raise RehearsalError("TRIGGER_EVENT_INVALID")
    try:
        row = _git_read(repository_path, "show", "-s", "--format=%H %P", commit_sha)
        parts = row.decode("ascii").strip().split()
        if len(parts) != 2 or parts[0] != commit_sha or _FULL_REF_SHA.fullmatch(parts[1]) is None:
            raise RehearsalError("TRIGGER_PARENT_INVALID")
        parent_sha = parts[1]
        changes = _git_read(
            repository_path,
            "diff-tree", "--no-commit-id", "--name-status", "-r", "--no-renames",
            parent_sha, commit_sha,
        ).decode("utf-8").splitlines()
        if len(changes) != 1 or not changes[0].startswith("A\t"):
            raise RehearsalError("TRIGGER_DIFF_INVALID")
        request_path = changes[0][2:]
        if not request_path.startswith("lifecycle-requests/") or "/" in request_path.removeprefix(
            "lifecycle-requests/"
        ):
            raise RehearsalError("TRIGGER_DIFF_INVALID")
        raw = _git_read(repository_path, "show", f"{commit_sha}:{request_path}")
        request = validate_route_request_bytes(raw)
        workflow_blob = _git_read(
            repository_path, "rev-parse", f"{parent_sha}:.github/workflows/cyax-lifecycle-dispatch.yml"
        ).decode("ascii").strip()
        tree = _git_read(repository_path, "rev-parse", f"{parent_sha}^{{tree}}").decode(
            "ascii"
        ).strip()
        paths = _git_read(
            repository_path, "ls-tree", "-r", "--name-only", parent_sha,
            "--", "scripts/version_lifecycle",
        ).decode("utf-8").splitlines()
        closure: dict[str, str] = {}
        for path in paths:
            if path.endswith(".py"):
                closure[path] = _git_read(
                    repository_path, "rev-parse", f"{parent_sha}:{path}"
                ).decode("ascii").strip()
        if (
            request_path != f"lifecycle-requests/{request.request_id}.json"
            or ref != f"refs/heads/rehearsal/trigger/{request.request_id}"
            or request.parent_sha != parent_sha
            or request.commit_sha
            or request.workflow_blob_sha != workflow_blob
            or request.request["plan"]["candidate_tree"] != tree
            or request.called_code_blobs != closure
        ):
            raise RehearsalError("TRIGGER_SOURCE_BINDING_INVALID")
        return ValidatedRouteRequest(
            request=request.request,
            request_bytes=request.request_bytes,
            request_sha256=request.request_sha256,
            plan_sha256=request.plan_sha256,
            owner_attestation_sha256=request.owner_attestation_sha256,
            request_id=request.request_id,
            request_path=request_path,
            commit_sha=commit_sha,
            parent_sha=parent_sha,
            workflow_blob_sha=workflow_blob,
            expected_vmm_sha=request.expected_vmm_sha,
            called_code_blobs=closure,
        )
    except GitHubProtectionError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError):
        raise RehearsalError("TRIGGER_SOURCE_UNAVAILABLE") from None


def _utc_seconds(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise RehearsalError("CLOCK_UNAVAILABLE")
    return current.astimezone(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


class AnonymousTransport(Protocol):
    def get(self, path: str) -> ApiResponse: ...


class AnonymousGitHubTransport:
    """Fixed-origin, direct TLS GET transport with no credential or proxy use."""

    def __init__(self) -> None:
        self.timeout = 10.0
        self.max_body = MAX_PROVIDER_BODY

    def get(self, path: str) -> ApiResponse:
        if not isinstance(path, str) or not _allowed_approval_path(path):
            raise RehearsalError("ANONYMOUS_PATH_NOT_ALLOWLISTED")
        connection = http.client.HTTPSConnection(
            "api.github.com", 443, timeout=self.timeout,
            context=ssl.create_default_context(),
        )
        try:
            connection.request(
                "GET", path,
                headers={
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": APPROVAL_API_VERSION,
                    "User-Agent": "CYAxiverse-approval-evidence-validator",
                },
            )
            response = connection.getresponse()
            body = response.read(self.max_body + 1)
            headers = {key.lower(): value for key, value in response.getheaders()}
            if len(body) > self.max_body:
                raise RehearsalError("APPROVAL_RESPONSE_OVERSIZE")
            return ApiResponse(response.status, body, headers)
        except RehearsalError:
            raise
        except Exception:
            raise RehearsalError("APPROVAL_READ_UNAVAILABLE") from None
        finally:
            connection.close()


def _allowed_approval_path(path: str) -> bool:
    prefix = f"/repos/{REPOSITORY}"
    if path == prefix:
        return True
    match = re.fullmatch(
        re.escape(prefix) + r"/actions/runs/([1-9][0-9]*)/attempts/1",
        path,
    )
    if match:
        return True
    match = re.fullmatch(
        re.escape(prefix) + r"/actions/runs/([1-9][0-9]*)/approvals",
        path,
    )
    if match:
        return True
    if re.fullmatch(
        re.escape(prefix)
        + r"/actions/runs/([1-9][0-9]*)/attempts/1/jobs\?per_page=100",
        path,
    ):
        return True
    if re.fullmatch(
        re.escape(prefix) + r"/issues/comments/[1-9][0-9]*", path
    ):
        return True
    if re.fullmatch(
        re.escape(prefix)
        + r"/git/ref/heads/rehearsal/(?:principal/cyax125-20261101-route|trigger/cyax125-p2-app-update|restricted/cyax125-20261101-move)",
        path,
    ):
        return True
    if path == f"{prefix}/git/commits/{_SEED_TARGET_SHA}":
        return True
    if path == (
        f"{prefix}/actions/runs?event=push&head_sha=a6df7c78e27ee81771cc06b0783a508edcaed320&per_page=100"
    ):
        return True
    return False


def _response_manifest(rows: list[tuple[str, ApiResponse]]) -> tuple[dict[str, Any], str]:
    manifest_rows = []
    for path, response in rows:
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
        manifest_rows.append({
            "path": path,
            "status": response.status,
            "content_type": headers.get("content-type"),
            "etag": headers.get("etag"),
            "request_id": headers.get("x-github-request-id"),
            "body_sha256": sha256_hex(response.body),
            "body_bytes": len(response.body),
        })
    manifest = {"responses": manifest_rows}
    return manifest, sha256_hex(_canonical(manifest))


def _json_response(response: ApiResponse) -> Any:
    if response.status != 200:
        raise RehearsalError("APPROVAL_PROVIDER_STATUS_INVALID")
    if not isinstance(response.body, bytes) or len(response.body) > MAX_PROVIDER_BODY:
        raise RehearsalError("APPROVAL_RESPONSE_OVERSIZE")
    headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    content_type = headers.get("content-type")
    if not isinstance(content_type, str) or content_type.split(";", 1)[0].strip().lower() not in {
        "application/json", "application/vnd.github+json"
    }:
        raise RehearsalError("APPROVAL_CONTENT_TYPE_INVALID")
    value = _strict_loads(response.body)
    if not isinstance(value, (dict, list)):
        raise RehearsalError("APPROVAL_RESPONSE_SHAPE_INVALID")
    if "link" in headers:
        raise RehearsalError("APPROVAL_PAGINATION_NOT_ALLOWED")
    return value


@dataclass(frozen=True, slots=True)
class VerifiedRunApproval:
    receipt: Mapping[str, Any]
    receipt_bytes: bytes = field(repr=False)
    receipt_sha256: str
    request_id: str
    plan_sha256: str
    run_id: int
    run_attempt: int
    head_sha: str
    job_id: int
    marker_sha256: str
    raw_responses: tuple[Mapping[str, Any], ...] = field(repr=False)
    _verification_token: object = field(repr=False, compare=False)


_APPROVAL_VERIFICATION_TOKEN = object()


def _parse_approval_marker(
    comment: Any,
    *,
    run_id: int,
    head_sha: str,
    request: ValidatedRouteRequest,
) -> tuple[dict[str, Any], bytes]:
    if not isinstance(comment, str) or not comment.startswith(APPROVAL_MARKER_PREFIX):
        raise RehearsalError("APPROVAL_MARKER_MISSING")
    raw = comment[len(APPROVAL_MARKER_PREFIX):].encode("utf-8")
    marker = _strict_loads(raw)
    if not isinstance(marker, dict) or not _schema_matches(marker, APPROVAL_MARKER_SCHEMA):
        raise RehearsalError("APPROVAL_MARKER_SCHEMA_INVALID")
    if _canonical(marker) != raw:
        raise RehearsalError("APPROVAL_MARKER_NOT_CANONICAL")
    if (
        marker["run_id"] != run_id
        or marker["run_attempt"] != 1
        or marker["repository"] != REPOSITORY
        or marker["head_sha"] != head_sha
        or marker["request_id"] != request.request_id
        or marker["plan_sha256"] != request.plan_sha256
        or marker["environment_id"] != ENVIRONMENT_ID
        or marker["environment_name"] != ENVIRONMENT_NAME
        or marker["owner_user_id"] != OWNER_ID
        or marker["no_admin_bypass_used"] is not True
    ):
        raise RehearsalError("APPROVAL_MARKER_BINDING_INVALID")
    return marker, raw


def _validate_repository_response(value: Any) -> None:
    if (
        not isinstance(value, dict)
        or type(value.get("id")) is not int
        or value.get("id") != REPOSITORY_ID
        or value.get("full_name") != REPOSITORY
        or value.get("private") is not False
    ):
        raise RehearsalError("APPROVAL_REPOSITORY_NOT_PUBLIC")


def _validate_run_response(value: Any, *, run_id: int, head_sha: str, ref: str) -> None:
    if not isinstance(value, dict):
        raise RehearsalError("APPROVAL_RUN_SHAPE_INVALID")
    repository = value.get("repository")
    actor = value.get("actor")
    workflow_path = value.get("path")
    if (
        type(value.get("id")) is not int
        or value.get("id") != run_id
        or type(value.get("run_attempt")) is not int
        or value.get("run_attempt") != 1
        or value.get("event") != "push"
        or value.get("head_sha") != head_sha
        or value.get("head_branch") != ref.removeprefix("refs/heads/")
        or not isinstance(workflow_path, str)
        or workflow_path.split("@", 1)[0]
        != ".github/workflows/cyax-lifecycle-dispatch.yml"
        or not isinstance(repository, dict)
        or type(repository.get("id")) is not int
        or repository.get("id") != REPOSITORY_ID
        or repository.get("full_name") != REPOSITORY
        or not isinstance(actor, dict)
        or type(actor.get("id")) is not int
        or actor.get("id") != OWNER_ID
    ):
        raise RehearsalError("APPROVAL_RUN_BINDING_INVALID")


def _validate_approval_rows(value: Any, *, run_id: int, head_sha: str,
                            request: ValidatedRouteRequest) -> tuple[bytes, str]:
    if not isinstance(value, list):
        raise RehearsalError("APPROVAL_HISTORY_SHAPE_INVALID")
    lifecycle_rows = []
    for row in value:
        if not isinstance(row, dict) or not isinstance(row.get("environments"), list):
            raise RehearsalError("APPROVAL_HISTORY_ROW_INVALID")
        matching_envs = [
            env for env in row["environments"]
            if isinstance(env, dict)
            and (env.get("id") == ENVIRONMENT_ID or env.get("name") == ENVIRONMENT_NAME)
        ]
        if matching_envs:
            lifecycle_rows.append((row, matching_envs))
    if len(lifecycle_rows) != 1:
        raise RehearsalError("APPROVAL_HISTORY_AMBIGUOUS")
    row, envs = lifecycle_rows[0]
    user = row.get("user")
    if (
        row.get("state") != "approved"
        or len(envs) != 1
        or type(envs[0].get("id")) is not int
        or envs[0].get("id") != ENVIRONMENT_ID
        or envs[0].get("name") != ENVIRONMENT_NAME
        or not isinstance(user, dict)
        or type(user.get("id")) is not int
        or user.get("id") != OWNER_ID
    ):
        raise RehearsalError("APPROVAL_HISTORY_OWNER_REVIEW_INVALID")
    _marker, raw = _parse_approval_marker(
        row.get("comment"), run_id=run_id, head_sha=head_sha, request=request
    )
    return raw, sha256_hex(raw)


def _validate_jobs_response(value: Any, *, run_id: int, head_sha: str) -> int:
    if (
        not isinstance(value, dict)
        or type(value.get("total_count")) is not int
        or value["total_count"] != 1
        or not isinstance(value.get("jobs"), list)
        or len(value["jobs"]) != 1
    ):
        raise RehearsalError("APPROVAL_JOB_SET_INVALID")
    job = value["jobs"][0]
    if not isinstance(job, dict):
        raise RehearsalError("APPROVAL_JOB_INVALID")
    if (
        type(job.get("id")) is not int
        or job["id"] <= 0
        or type(job.get("run_id")) is not int
        or job["run_id"] != run_id
        or job.get("name") != "Validate and fence an Owner rehearsal request"
        or job.get("head_sha") != head_sha
        or job.get("status") != "in_progress"
        or not isinstance(job.get("steps"), list)
    ):
        raise RehearsalError("APPROVAL_JOB_BINDING_INVALID")
    steps = job["steps"]
    expected = (
        "Fetch the exact trigger commit with anonymous read-only access",
        "Validate exact route and approval evidence before App mint",
        "Mint a scoped token and execute the fixed rehearsal route",
    )
    if not isinstance(steps, list) or len(steps) < len(expected):
        raise RehearsalError("APPROVAL_JOB_STEPS_INVALID")
    offset = 0
    if steps and isinstance(steps[0], dict) and steps[0].get("name") == "Set up job":
        setup = steps[0]
        if setup.get("status") != "completed" or setup.get("conclusion") != "success":
            raise RehearsalError("APPROVAL_RUNNER_SETUP_NOT_COMPLETE")
        offset = 1
    has_complete_step = (
        len(steps) > offset + len(expected)
        and isinstance(steps[-1], dict)
        and steps[-1].get("name") == "Complete job"
    )
    expected_length = offset + len(expected) + (1 if has_complete_step else 0)
    if len(steps) != expected_length:
        raise RehearsalError("APPROVAL_JOB_STEPS_INVALID")
    for index, (step, name) in enumerate(zip(steps[offset:offset + len(expected)], expected)):
        if not isinstance(step, dict) or step.get("name") != name:
            raise RehearsalError("APPROVAL_JOB_STEPS_INVALID")
        status = step.get("status")
        if index == 0 and (status != "completed" or step.get("conclusion") != "success"):
            raise RehearsalError("APPROVAL_FETCH_STEP_NOT_COMPLETE")
        if index == 1 and status != "in_progress":
            raise RehearsalError("APPROVAL_VALIDATOR_STEP_NOT_CURRENT")
        if index == 2 and (status != "queued" or step.get("conclusion") is not None):
            raise RehearsalError("APPROVAL_SECRET_STEP_ALREADY_STARTED")
    if has_complete_step:
        complete = steps[-1]
        if complete.get("status") != "queued" or complete.get("conclusion") is not None:
            raise RehearsalError("APPROVAL_RUNNER_COMPLETION_ALREADY_STARTED")
    return job["id"]


def verify_run_approval(
    request: ValidatedRouteRequest,
    *,
    run_id: int,
    run_attempt: int,
    head_sha: str,
    ref: str,
    transport: AnonymousTransport | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> VerifiedRunApproval:
    """Read the exact public run/approval/job binding twice, with no auth."""

    if (
        not isinstance(request, ValidatedRouteRequest)
        or type(run_id) is not int or run_id <= 0
        or type(run_attempt) is not int or run_attempt != 1
        or head_sha != request.commit_sha
        or ref != f"refs/heads/rehearsal/trigger/{request.request_id}"
    ):
        raise RehearsalError("APPROVAL_CONTEXT_INVALID")
    reader = transport or AnonymousGitHubTransport()
    run_prefix = f"/repos/{REPOSITORY}/actions/runs/{run_id}"
    paths = (
        f"/repos/{REPOSITORY}",
        f"{run_prefix}/attempts/1",
        f"{run_prefix}/approvals",
        f"{run_prefix}/attempts/1/jobs?per_page=100",
    )
    started_utc = _utc_seconds(utc_now())
    started = monotonic()
    all_rows: list[list[tuple[str, ApiResponse]]] = []
    for _round in range(2):
        rows: list[tuple[str, ApiResponse]] = []
        for path in paths:
            if not _allowed_approval_path(path):
                raise RehearsalError("APPROVAL_PATH_NOT_ALLOWLISTED")
            rows.append((path, reader.get(path)))
            if monotonic() - started > MAX_WINDOW_SECONDS:
                raise RehearsalError("APPROVAL_READ_WINDOW_EXCEEDED")
        all_rows.append(rows)
        if monotonic() - started > MAX_WINDOW_SECONDS:
            raise RehearsalError("APPROVAL_READ_WINDOW_EXCEEDED")
    finished_utc = _utc_seconds(utc_now())
    if monotonic() - started > MAX_WINDOW_SECONDS:
        raise RehearsalError("APPROVAL_READ_WINDOW_EXCEEDED")
    first_values = [_json_response(response) for _path, response in all_rows[0]]
    second_values = [_json_response(response) for _path, response in all_rows[1]]
    if any(
        first.body != second.body or first.status != second.status
        or {
            "etag": {str(k).lower(): str(v) for k, v in first.headers.items()}.get("etag"),
            "content-type": {str(k).lower(): str(v) for k, v in first.headers.items()}.get("content-type"),
        }
        != {
            "etag": {str(k).lower(): str(v) for k, v in second.headers.items()}.get("etag"),
            "content-type": {str(k).lower(): str(v) for k, v in second.headers.items()}.get("content-type"),
        }
        for (_p1, first), (_p2, second) in zip(all_rows[0], all_rows[1])
    ):
        raise RehearsalError("APPROVAL_READS_CHANGED")
    for values in (first_values, second_values):
        _validate_repository_response(values[0])
        _validate_run_response(values[1], run_id=run_id, head_sha=head_sha, ref=ref)
        _validate_approval_rows(values[2], run_id=run_id, head_sha=head_sha, request=request)
        current_job_id = _validate_jobs_response(values[3], run_id=run_id, head_sha=head_sha)
    marker_raw, marker_sha = _validate_approval_rows(
        first_values[2], run_id=run_id, head_sha=head_sha, request=request
    )
    first_manifest, first_sha = _response_manifest(all_rows[0])
    second_manifest, second_sha = _response_manifest(all_rows[1])
    if current_job_id <= 0:
        raise RehearsalError("APPROVAL_JOB_ID_INVALID")
    receipt = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "run_id": run_id,
        "run_attempt": 1,
        "head_sha": head_sha,
        "request_id": request.request_id,
        "plan_sha256": request.plan_sha256,
        "job_id": current_job_id,
        "environment_id": ENVIRONMENT_ID,
        "owner_user_id": OWNER_ID,
        "approval_state": "approved",
        "approval_event_time": None,
        "approval_time_evidence": "UNKNOWN_NOT_DOCUMENTED",
        "admin_bypass_evidence": "OWNER_PROCEDURAL_ATTESTATION_NOT_PROVIDER_VERIFIED",
        "marker_sha256": marker_sha,
        "first_response_manifest_sha256": first_sha,
        "second_response_manifest_sha256": second_sha,
        "collection_started_at": started_utc,
        "collection_finished_at": finished_utc,
        "status": "APPROVAL_BINDING_VERIFIED_WITH_EXPLICIT_LIMITS",
        "failure_reason": None,
    }
    raw_responses = tuple(
        {
            "path": path,
            "status": response.status,
            "headers": {str(key).lower(): str(value) for key, value in response.headers.items()},
            "body": response.body,
            "body_sha256": sha256_hex(response.body),
        }
        for rows in all_rows for path, response in rows
    )
    # The response manifests commit request IDs and available headers; retain
    # both raw snapshots in memory for the workflow's evidence writer.
    _ = first_manifest, second_manifest, marker_raw
    receipt_bytes = _canonical(receipt)
    return VerifiedRunApproval(
        receipt=receipt,
        receipt_bytes=receipt_bytes,
        receipt_sha256=sha256_hex(receipt_bytes),
        request_id=request.request_id,
        plan_sha256=request.plan_sha256,
        run_id=run_id,
        run_attempt=1,
        head_sha=head_sha,
        job_id=current_job_id,
        marker_sha256=marker_sha,
        raw_responses=raw_responses,
        _verification_token=_APPROVAL_VERIFICATION_TOKEN,
    )


def validate_approval_receipt(
    raw: bytes,
    request: ValidatedRouteRequest,
    *,
    run_id: int,
    run_attempt: int,
    head_sha: str,
) -> VerifiedRunApproval:
    """Revalidate the closed first-step receipt before any App-token use."""

    if not isinstance(request, ValidatedRouteRequest):
        raise RehearsalError("APPROVAL_RECEIPT_REQUEST_INVALID")
    if not isinstance(raw, bytes) or not raw or len(raw) > 16384:
        raise RehearsalError("APPROVAL_RECEIPT_SIZE_INVALID")
    receipt = _strict_loads(raw)
    required = {
        "schema_version", "repository", "run_id", "run_attempt", "head_sha",
        "request_id", "plan_sha256", "job_id", "environment_id", "owner_user_id",
        "approval_state", "approval_event_time", "approval_time_evidence",
        "admin_bypass_evidence", "marker_sha256", "first_response_manifest_sha256",
        "second_response_manifest_sha256", "collection_started_at",
        "collection_finished_at", "status", "failure_reason",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise RehearsalError("APPROVAL_RECEIPT_FIELDS_INVALID")
    for key in ("run_id", "run_attempt", "job_id", "environment_id", "owner_user_id", "schema_version"):
        if type(receipt[key]) is not int:
            raise RehearsalError("APPROVAL_RECEIPT_TYPE_INVALID")
    for key in ("first_response_manifest_sha256", "second_response_manifest_sha256", "marker_sha256"):
        if not isinstance(receipt[key], str) or re.fullmatch(r"[0-9a-f]{64}", receipt[key]) is None:
            raise RehearsalError("APPROVAL_RECEIPT_DIGEST_INVALID")
    for key in ("collection_started_at", "collection_finished_at"):
        value = receipt[key]
        try:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        except (TypeError, ValueError):
            raise RehearsalError("APPROVAL_COLLECTION_TIME_INVALID") from None
        if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
            raise RehearsalError("APPROVAL_COLLECTION_TIME_INVALID")
    if (
        _canonical(receipt) != raw
        or receipt["schema_version"] != 1
        or receipt["repository"] != REPOSITORY
        or receipt["run_id"] != run_id
        or type(run_id) is not int
        or type(run_attempt) is not int
        or run_attempt != 1
        or receipt["run_attempt"] != run_attempt
        or receipt["head_sha"] != head_sha
        or request.commit_sha != head_sha
        or receipt["request_id"] != request.request_id
        or receipt["plan_sha256"] != request.plan_sha256
        or receipt["environment_id"] != ENVIRONMENT_ID
        or receipt["owner_user_id"] != OWNER_ID
        or receipt["approval_state"] != "approved"
        or receipt["approval_event_time"] is not None
        or receipt["approval_time_evidence"] != "UNKNOWN_NOT_DOCUMENTED"
        or receipt["admin_bypass_evidence"] != "OWNER_PROCEDURAL_ATTESTATION_NOT_PROVIDER_VERIFIED"
        or receipt["status"] != "APPROVAL_BINDING_VERIFIED_WITH_EXPLICIT_LIMITS"
        or receipt["failure_reason"] is not None
        or receipt["job_id"] <= 0
        or receipt["collection_finished_at"] < receipt["collection_started_at"]
    ):
        raise RehearsalError("APPROVAL_RECEIPT_BINDING_INVALID")
    return VerifiedRunApproval(
        receipt=receipt,
        receipt_bytes=raw,
        receipt_sha256=sha256_hex(raw),
        request_id=request.request_id,
        plan_sha256=request.plan_sha256,
        run_id=run_id,
        run_attempt=1,
        head_sha=head_sha,
        job_id=receipt["job_id"],
        marker_sha256=receipt["marker_sha256"],
        raw_responses=(),
        _verification_token=_APPROVAL_VERIFICATION_TOKEN,
    )


def serialize_approval_envelope(approval: VerifiedRunApproval) -> bytes:
    """Serialize the fixed nonsecret response snapshots for same-job handoff."""

    if (
        not isinstance(approval, VerifiedRunApproval)
        or approval._verification_token is not _APPROVAL_VERIFICATION_TOKEN
        or sha256_hex(approval.receipt_bytes) != approval.receipt_sha256
        or len(approval.raw_responses) != 8
    ):
        raise RehearsalError("APPROVAL_ENVELOPE_SOURCE_INVALID")
    responses: list[dict[str, Any]] = []
    for row in approval.raw_responses:
        if not isinstance(row, Mapping):
            raise RehearsalError("APPROVAL_ENVELOPE_SOURCE_INVALID")
        path, status, headers, body, body_sha256 = (
            row.get("path"), row.get("status"), row.get("headers"),
            row.get("body"), row.get("body_sha256"),
        )
        if (
            not isinstance(path, str)
            or type(status) is not int
            or status != 200
            or not isinstance(headers, Mapping)
            or not isinstance(body, bytes)
            or len(body) > MAX_PROVIDER_BODY
            or body_sha256 != sha256_hex(body)
            or any(
                not isinstance(key, str) or key != key.lower()
                or not isinstance(value, str)
                for key, value in headers.items()
            )
            or any(key in headers for key in (
                "authorization", "proxy-authorization", "set-cookie"
            ))
            or len(headers) > 64
            or sum(len(key) + len(value) for key, value in headers.items()) > 16384
        ):
            raise RehearsalError("APPROVAL_ENVELOPE_SOURCE_INVALID")
        responses.append({
            "path": path,
            "status": status,
            "headers": dict(headers),
            "body_b64": base64.b64encode(body).decode("ascii"),
            "body_sha256": body_sha256,
        })
    return _canonical({
        "schema_version": 1,
        "receipt_b64": base64.b64encode(approval.receipt_bytes).decode("ascii"),
        "receipt_sha256": approval.receipt_sha256,
        "responses": responses,
    })


def rehydrate_approval_envelope(
    raw_envelope: bytes,
    request: ValidatedRouteRequest,
    *,
    run_id: int,
    run_attempt: int,
    head_sha: str,
) -> VerifiedRunApproval:
    """Rebuild approval provenance from the closed same-job envelope."""

    if not isinstance(raw_envelope, bytes) or not raw_envelope or len(raw_envelope) > 12 * 1024 * 1024:
        raise RehearsalError("APPROVAL_ENVELOPE_SIZE_INVALID")
    envelope = _strict_loads(raw_envelope)
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"schema_version", "receipt_b64", "receipt_sha256", "responses"}
        or type(envelope.get("schema_version")) is not int
        or envelope["schema_version"] != 1
        or _canonical(envelope) != raw_envelope
        or not isinstance(envelope.get("receipt_b64"), str)
        or not isinstance(envelope.get("receipt_sha256"), str)
        or not isinstance(envelope.get("responses"), list)
        or len(envelope["responses"]) != 8
    ):
        raise RehearsalError("APPROVAL_ENVELOPE_INVALID")
    try:
        receipt_bytes = base64.b64decode(envelope["receipt_b64"], validate=True)
    except (ValueError, TypeError):
        raise RehearsalError("APPROVAL_ENVELOPE_RECEIPT_INVALID") from None
    if sha256_hex(receipt_bytes) != envelope["receipt_sha256"]:
        raise RehearsalError("APPROVAL_ENVELOPE_RECEIPT_DIGEST_MISMATCH")
    approval = validate_approval_receipt(
        receipt_bytes, request, run_id=run_id,
        run_attempt=run_attempt, head_sha=head_sha,
    )
    run_prefix = f"/repos/{REPOSITORY}/actions/runs/{run_id}"
    expected_paths = (
        f"/repos/{REPOSITORY}",
        f"{run_prefix}/attempts/1",
        f"{run_prefix}/approvals",
        f"{run_prefix}/attempts/1/jobs?per_page=100",
    ) * 2
    rows: list[tuple[str, ApiResponse]] = []
    raw_responses: list[dict[str, Any]] = []
    total_body_bytes = 0
    for expected_path, item in zip(expected_paths, envelope["responses"]):
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "status", "headers", "body_b64", "body_sha256"}
            or item.get("path") != expected_path
            or not _allowed_approval_path(expected_path)
            or type(item.get("status")) is not int
            or item["status"] != 200
            or not isinstance(item.get("headers"), dict)
            or not isinstance(item.get("body_b64"), str)
            or not isinstance(item.get("body_sha256"), str)
        ):
            raise RehearsalError("APPROVAL_ENVELOPE_RESPONSE_INVALID")
        headers = item["headers"]
        if (
            any(
                not isinstance(key, str) or key != key.lower()
                or not isinstance(value, str)
                for key, value in headers.items()
            )
            or any(key in headers for key in (
                "authorization", "proxy-authorization", "set-cookie"
            ))
            or len(headers) > 64
            or sum(len(key) + len(value) for key, value in headers.items()) > 16384
        ):
            raise RehearsalError("APPROVAL_ENVELOPE_RESPONSE_INVALID")
        try:
            body = base64.b64decode(item["body_b64"], validate=True)
        except (ValueError, TypeError):
            raise RehearsalError("APPROVAL_ENVELOPE_BODY_INVALID") from None
        if len(body) > MAX_PROVIDER_BODY or item["body_sha256"] != sha256_hex(body):
            raise RehearsalError("APPROVAL_ENVELOPE_BODY_DIGEST_MISMATCH")
        total_body_bytes += len(body)
        response = ApiResponse(item["status"], body, dict(headers))
        _json_response(response)
        rows.append((expected_path, response))
        raw_responses.append({
            "path": expected_path,
            "status": item["status"],
            "headers": dict(headers),
            "body": body,
            "body_sha256": item["body_sha256"],
        })
    if total_body_bytes > 8 * MAX_PROVIDER_BODY:
        raise RehearsalError("APPROVAL_ENVELOPE_OVERSIZE")
    first_rows, second_rows = rows[:4], rows[4:]
    for (first_path, first), (second_path, second) in zip(first_rows, second_rows):
        if (
            first_path != second_path
            or first.body != second.body
            or first.status != second.status
            or {
                key: {str(k).lower(): str(v) for k, v in first.headers.items()}.get(key)
                for key in ("etag", "content-type")
            }
            != {
                key: {str(k).lower(): str(v) for k, v in second.headers.items()}.get(key)
                for key in ("etag", "content-type")
            }
        ):
            raise RehearsalError("APPROVAL_READS_CHANGED")
    first_values = [_json_response(response) for _path, response in first_rows]
    second_values = [_json_response(response) for _path, response in second_rows]
    observed_jobs: list[int] = []
    observed_markers: list[str] = []
    for values in (first_values, second_values):
        _validate_repository_response(values[0])
        _validate_run_response(values[1], run_id=run_id, head_sha=head_sha,
                               ref=f"refs/heads/rehearsal/trigger/{request.request_id}")
        _marker_raw, marker_sha = _validate_approval_rows(
            values[2], run_id=run_id, head_sha=head_sha, request=request,
        )
        observed_markers.append(marker_sha)
        observed_jobs.append(_validate_jobs_response(values[3], run_id=run_id, head_sha=head_sha))
    _first_manifest, first_sha = _response_manifest(first_rows)
    _second_manifest, second_sha = _response_manifest(second_rows)
    if (
        observed_jobs != [approval.job_id, approval.job_id]
        or observed_markers != [approval.marker_sha256, approval.marker_sha256]
        or first_sha != approval.receipt["first_response_manifest_sha256"]
        or second_sha != approval.receipt["second_response_manifest_sha256"]
    ):
        raise RehearsalError("APPROVAL_ENVELOPE_RECEIPT_BINDING_INVALID")
    return replace(approval, raw_responses=tuple(raw_responses))


@dataclass(frozen=True, slots=True)
class CaseObservation:
    status: str
    evidence: Mapping[str, Any]
    provider_status: int | None = None
    failure_reason: str | None = None
    p5_subresult: Mapping[str, Any] | None = None


class RehearsalAdapter(Protocol):
    def preflight_route(self, request: ValidatedRouteRequest,
                        approval: VerifiedRunApproval) -> CaseObservation: ...
    def case_lock(self, ref: str) -> Any: ...
    def verify_rehearsal_freeze_epoch(self, request: ValidatedRouteRequest,
                                      case_id: str, phase: str) -> Any: ...
    def execute_case(self, case_id: str, request: ValidatedRouteRequest,
                     approval: VerifiedRunApproval, proof: Any) -> CaseObservation: ...


class AppendOnlyJournal(Protocol):
    def claim_case(self, request: ValidatedRouteRequest, case_id: str,
                   proof: Any) -> str: ...
    def append_case_result(self, result: Mapping[str, Any],
                           evidence: Mapping[str, Any]) -> str: ...


def _status_next(case_id: str, status: str) -> tuple[str | None, bool]:
    if status in {
        "BLOCKED_UNKNOWN", "BLOCKED_NOT_VERIFIABLE", "BLOCKED_UNEXPECTED_ACCEPTANCE",
        "STOP_OWNER_ACTION_REQUIRED", "NOT_EXECUTED",
    }:
        return None, True
    if status not in {
        "PASS_ACCEPTED", "PASS_DENIED_TARGET_CONTROL", "PASS_OWNER_ROUTE_NO_PUSH",
    }:
        return None, True
    index = _CASE_INDEX[case_id]
    if case_id == "P5-event-observe":
        return "P5-duplicate-fence", False
    if case_id == "P5-duplicate-fence":
        return "4g", False
    if index + 1 == len(CASE_ORDER):
        return None, False
    return CASE_ORDER[index + 1], False


def validate_driver_result(value: Any) -> None:
    if not isinstance(value, dict) or not _schema_matches(value, DRIVER_RETURN_SCHEMA):
        raise RehearsalError("DRIVER_RETURN_SCHEMA_INVALID")
    status = value["status"]
    expected_next, unresolved = _status_next(value["case_id"], status)
    if value["next_case"] != expected_next or value["unresolved"] is not unresolved:
        raise RehearsalError("DRIVER_RETURN_ADJACENCY_INVALID")
    sub = value["p5_subresult"]
    if value["case_id"] == "P5-event-observe":
        if sub is None:
            raise RehearsalError("P5_EVENT_SUBRESULT_REQUIRED")
        counts = [sub.get(name) for name in (
            "second_driver_dispatch_count", "api_mutation_count", "git_push_count"
        )]
        if status == "PASS_ACCEPTED" and (
            any(type(count) is not int or count != 0 for count in counts)
            or sub.get("failure_reason") is not None
            or sub.get("matching_app_trigger_guard_exercised") is not False
            or (
                sub.get("fence_state") == "UNKNOWN"
                and type(sub.get("observed_off_ref_event")) is not bool
            )
            or value["next_case"] != "P5-duplicate-fence"
            or value["unresolved"] is not False
        ):
            raise RehearsalError("P5_EVENT_PASS_EFFECTS_INVALID")
        if status != "PASS_ACCEPTED" and (
            value["next_case"] is not None or value["unresolved"] is not True
        ):
            raise RehearsalError("P5_EVENT_FAILURE_ADJACENCY_INVALID")
    if value["case_id"] == "P5-duplicate-fence" and status == "PASS_DENIED_TARGET_CONTROL":
        counts = [sub.get(name) for name in (
            "second_driver_dispatch_count", "api_mutation_count", "git_push_count"
        )]
        if (
            any(type(count) is not int or count != 0 for count in counts)
            or sub.get("fence_state") != "VERIFIED_MATCH"
            or sub.get("request_plan_match") is not True
            or not isinstance(sub.get("existing_fence_sha"), str)
            or sub.get("failure_reason") is not None
        ):
            raise RehearsalError("P5_DUPLICATE_PASS_EFFECTS_INVALID")


def _result(
    request: ValidatedRouteRequest,
    case_id: str,
    status: str,
    evidence: Mapping[str, Any],
    *,
    p5_subresult: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    next_case, unresolved = _status_next(case_id, status)
    digest = sha256_hex(_canonical(dict(evidence)))
    result = {
        "schema_version": 1,
        "request_id": request.request_id,
        "plan_sha256": request.plan_sha256,
        "case_id": case_id,
        "status": status,
        "evidence_sha256": digest,
        "next_case": next_case,
        "unresolved": unresolved,
        "p5_subresult": p5_subresult if case_id in {
            "P5-event-observe", "P5-duplicate-fence"
        } else None,
    }
    validate_driver_result(result)
    return result


def _p5_unknown(case_id: str, reason: str) -> Mapping[str, Any] | None:
    if case_id not in {"P5-event-observe", "P5-duplicate-fence"}:
        return None
    return {
        "matching_app_trigger_guard_exercised": False,
        "observed_off_ref_event": None,
        "second_driver_dispatch_count": None,
        "api_mutation_count": None,
        "git_push_count": None,
        "existing_fence_sha": None,
        "fence_state": "UNKNOWN",
        "request_plan_match": None,
        "failure_reason": reason,
    }


def _expected_final_case_state(
    operation: Mapping[str, Any], proof: Any, observation: CaseObservation
) -> tuple[dict[str, Any] | None, str | None]:
    """Derive the exact state required for any result that can advance."""

    if operation.get("case", "").startswith("4d-") and observation.status in {
        "PASS_ACCEPTED", "STOP_OWNER_ACTION_REQUIRED",
    }:
        return None, "WORKFLOWS_CREATE_DISCRIMINATOR_UNQUALIFIED"
    expected_result = operation.get("expected_result")
    if observation.status == "PASS_ACCEPTED":
        if expected_result not in {
            "ACCEPT_EXACT_FF",
            "ACCEPT_OR_WORKFLOW_CAPABILITY_STOP",
            "READ_ONLY_CORROBORATION_NOT_TRIGGER_GUARD_PASS",
        }:
            return None, "CASE_TERMINAL_RESULT_CONTRACT_MISMATCH"
        expected_sha = operation.get("target")
    elif observation.status == "PASS_DENIED_TARGET_CONTROL":
        if expected_result not in {
            "DENY_TARGET_RULE", "REQUEST_ID_ALREADY_USED_NO_SECOND_DISPATCH",
        }:
            return None, "CASE_TERMINAL_RESULT_CONTRACT_MISMATCH"
        expected_sha = getattr(proof, "ref_sha", None)
    elif observation.status == "PASS_OWNER_ROUTE_NO_PUSH":
        if expected_result != "OWNER_ROUTE_REQUIRED_NO_PUSH":
            return None, "CASE_TERMINAL_RESULT_CONTRACT_MISMATCH"
        expected_sha = getattr(proof, "ref_sha", None)
    elif observation.status == "STOP_OWNER_ACTION_REQUIRED":
        if expected_result not in {
            "STOP_OWNER_ACTION_REQUIRED", "ACCEPT_OR_WORKFLOW_CAPABILITY_STOP",
        }:
            return None, "CASE_TERMINAL_RESULT_CONTRACT_MISMATCH"
        expected_sha = getattr(proof, "ref_sha", None)
    else:
        return None, None
    return {
        "ref": getattr(proof, "ref", None),
        "object_type": "commit" if expected_sha is not None else None,
        "sha": expected_sha,
    }, None


def run_rehearsal(
    request: ValidatedRouteRequest,
    approval: VerifiedRunApproval,
    adapter: RehearsalAdapter,
    journal: AppendOnlyJournal,
) -> dict[str, Any]:
    """Run the one fixed 18-case sequence; no caller-selected start/resume."""

    if (
        not isinstance(request, ValidatedRouteRequest)
        or not isinstance(approval, VerifiedRunApproval)
        or approval._verification_token is not _APPROVAL_VERIFICATION_TOKEN
        or approval.request_id != request.request_id
        or approval.plan_sha256 != request.plan_sha256
        or approval.head_sha != request.commit_sha
        or request.request["purpose"] != "gate-a-route-proof"
    ):
        raise RehearsalError("DRIVER_ENTRY_BINDING_INVALID")
    if any(not callable(getattr(adapter, name, None)) for name in (
        "preflight_route", "case_lock", "verify_rehearsal_freeze_epoch", "execute_case",
    )) or any(not callable(getattr(journal, name, None)) for name in (
        "claim_case", "append_case_result",
    )):
        raise RehearsalError("DRIVER_INTERFACE_INVALID")
    try:
        parsed_request = validate_route_request_bytes(request.request_bytes)
    except RehearsalError:
        raise RehearsalError("DRIVER_REQUEST_REVALIDATION_FAILED") from None
    if (
        request.request_sha256 != parsed_request.request_sha256
        or request.plan_sha256 != parsed_request.plan_sha256
        or request.owner_attestation_sha256 != parsed_request.owner_attestation_sha256
        or request.request_id != parsed_request.request_id
        or request.request_path != parsed_request.request_path
        or request.parent_sha != parsed_request.parent_sha
        or request.workflow_blob_sha != parsed_request.workflow_blob_sha
        or request.expected_vmm_sha != parsed_request.expected_vmm_sha
        or dict(request.called_code_blobs) != dict(parsed_request.called_code_blobs)
        or not isinstance(request.commit_sha, str)
        or _FULL_REF_SHA.fullmatch(request.commit_sha) is None
    ):
        raise RehearsalError("DRIVER_REQUEST_BINDING_INVALID")
    request = replace(
        parsed_request,
        commit_sha=request.commit_sha,
        parent_sha=request.parent_sha,
        workflow_blob_sha=request.workflow_blob_sha,
        called_code_blobs=dict(request.called_code_blobs),
    )
    try:
        approval = rehydrate_approval_envelope(
            serialize_approval_envelope(approval), request,
            run_id=approval.run_id, run_attempt=approval.run_attempt,
            head_sha=approval.head_sha,
        )
    except RehearsalError:
        raise RehearsalError("DRIVER_APPROVAL_REVALIDATION_FAILED") from None
    preflight = adapter.preflight_route(request, approval)
    if preflight.status != "PASS_ACCEPTED":
        case_id = "P2-app-create"
        return _result(
            request, case_id,
            preflight.status if preflight.status in {
                "STOP_OWNER_ACTION_REQUIRED", "BLOCKED_UNKNOWN", "BLOCKED_NOT_VERIFIABLE"
            } else "BLOCKED_NOT_VERIFIABLE",
            preflight.evidence,
        )
    for case_id in CASE_ORDER:
        operation = OPERATION_TABLE[_CASE_INDEX[case_id]]
        ref = operation["ref"]
        if ref.startswith("DETERMINISTIC "):
            ref = f"refs/heads/rehearsal/lifecycle/requests/{request.request_id}"
        with adapter.case_lock(ref):
            try:
                proof = adapter.verify_rehearsal_freeze_epoch(
                    request, case_id, "pre"
                )
                if proof is None:
                    raise RehearsalError("FREEZE_EPOCH_PROOF_MISSING")
            except Exception as error:
                reason = error.reason if isinstance(error, RehearsalError) else getattr(
                    error, "reason_code", "CASE_PRECONDITION_UNKNOWN"
                )
                observation = CaseObservation(
                    "BLOCKED_NOT_VERIFIABLE", {"reason": reason}, failure_reason=reason
                )
                evidence = dict(observation.evidence)
                evidence.setdefault("request_sha256", request.request_sha256)
                evidence.setdefault("plan_sha256", request.plan_sha256)
                evidence.setdefault("approval_receipt_sha256", approval.receipt_sha256)
                evidence.setdefault("failure_reason", observation.failure_reason)
                return _result(
                    request, case_id, observation.status, evidence,
                    p5_subresult=_p5_unknown(case_id, reason),
                )
            try:
                journal.claim_case(request, case_id, proof)
            except Exception as error:
                reason = getattr(error, "reason_code", "CASE_CLAIM_WRITE_UNKNOWN")
                # A claim may have reached the provider. Do not retry it or
                # append a compensating result ref after an unknown outcome.
                return _result(
                    request, case_id, "BLOCKED_UNKNOWN",
                    {
                        "reason": reason,
                        "request_sha256": request.request_sha256,
                        "plan_sha256": request.plan_sha256,
                        "approval_receipt_sha256": approval.receipt_sha256,
                    },
                    p5_subresult=_p5_unknown(case_id, reason),
                )
            unreconciled_postproof: dict[str, Any] | None = None
            try:
                observation = adapter.execute_case(
                    case_id, request, approval, proof
                )
            except Exception as error:
                reason = error.reason if isinstance(error, RehearsalError) else getattr(
                    error, "reason_code", "CASE_EXECUTION_UNKNOWN"
                )
                unknown = any(token in str(reason) for token in (
                    "UNKNOWN", "UNRECONCILED", "INFLIGHT", "WRITE",
                ))
                observation = CaseObservation(
                    "BLOCKED_UNKNOWN" if unknown else "BLOCKED_NOT_VERIFIABLE",
                    {"reason": reason, "request_sha256": request.request_sha256,
                     "plan_sha256": request.plan_sha256,
                     "approval_receipt_sha256": approval.receipt_sha256,
                     "pre": getattr(proof, "to_dict", lambda: {})()},
                    failure_reason=reason,
                    p5_subresult=_p5_unknown(case_id, reason),
                )
                try:
                    postproof = adapter.verify_rehearsal_freeze_epoch(
                        request, case_id, "post"
                    )
                    postproof_record = getattr(postproof, "to_dict", lambda: {})()
                    generation_matches = (
                        postproof is not None
                        and getattr(postproof, "generation", None)
                        == getattr(proof, "generation", None)
                    )
                    ref_matches = (
                        postproof is not None
                        and getattr(postproof, "ref", None) == getattr(proof, "ref", None)
                    )
                    final_reconciled = generation_matches and ref_matches
                except Exception as post_error:
                    post_reason = getattr(post_error, "reason_code", "FINAL_POSTPROOF_UNAVAILABLE")
                    postproof_record = {"state": "UNKNOWN", "reason": post_reason}
                    final_reconciled = False
                observation = CaseObservation(
                    "BLOCKED_UNKNOWN",
                    {
                        **dict(observation.evidence),
                        "pre": getattr(proof, "to_dict", lambda: {})(),
                        "post": postproof_record,
                        "final_postproof_reconciled": final_reconciled,
                    },
                    failure_reason=observation.failure_reason or "CASE_EXECUTION_UNKNOWN",
                    p5_subresult=observation.p5_subresult,
                )
                unreconciled_postproof = dict(observation.evidence)
            else:
                try:
                    postproof = adapter.verify_rehearsal_freeze_epoch(
                        request, case_id, "post"
                    )
                    postproof_record = getattr(postproof, "to_dict", lambda: {})()
                except Exception as error:
                    reason = error.reason if isinstance(error, RehearsalError) else getattr(
                        error, "reason_code", "FINAL_POSTPROOF_UNAVAILABLE"
                    )
                    observation = CaseObservation(
                        "BLOCKED_UNKNOWN",
                        {"reason": "FINAL_POSTPROOF_UNAVAILABLE",
                         "postproof_error": reason,
                         "operation_evidence": dict(observation.evidence),
                         "pre": getattr(proof, "to_dict", lambda: {})(),
                         "post": {"state": "UNKNOWN"}},
                        failure_reason="FINAL_POSTPROOF_UNAVAILABLE",
                        p5_subresult=observation.p5_subresult,
                    )
                    unreconciled_postproof = dict(observation.evidence)
                else:
                    expected_state, contract_error = _expected_final_case_state(
                        operation, proof, observation
                    )
                    generation_matches = (
                        postproof is not None
                        and getattr(postproof, "generation", None)
                        == getattr(proof, "generation", None)
                    )
                    ref_matches = (
                        postproof is not None
                        and getattr(postproof, "ref", None) == getattr(proof, "ref", None)
                    )
                    if observation.status in {
                        "PASS_ACCEPTED", "PASS_DENIED_TARGET_CONTROL",
                        "PASS_OWNER_ROUTE_NO_PUSH", "STOP_OWNER_ACTION_REQUIRED",
                    } and (
                        contract_error is not None
                        or expected_state is None
                        or not generation_matches
                        or not ref_matches
                        or getattr(postproof, "ref_sha", None) != expected_state["sha"]
                        or getattr(postproof, "ref_object_type", None)
                        != expected_state["object_type"]
                    ):
                        reason = contract_error or "FINAL_POSTPROOF_STATE_UNRECONCILED"
                        postproof_evidence = {
                            "reason": reason,
                            "expected_post_state": expected_state or {},
                            "operation_evidence": dict(observation.evidence),
                            "pre": getattr(proof, "to_dict", lambda: {})(),
                            "post": postproof_record or {"state": "UNKNOWN"},
                        }
                        observation = CaseObservation(
                            "BLOCKED_UNKNOWN", postproof_evidence,
                            failure_reason=reason,
                            p5_subresult=observation.p5_subresult,
                        )
                        unreconciled_postproof = postproof_evidence
                    elif postproof is None or not generation_matches or not ref_matches:
                        postproof_evidence = {
                            "reason": "FREEZE_EPOCH_CHANGED",
                            "operation_evidence": dict(observation.evidence),
                            "pre": getattr(proof, "to_dict", lambda: {})(),
                            "post": postproof_record or {"state": "UNKNOWN"},
                        }
                        observation = CaseObservation(
                            "BLOCKED_UNKNOWN", postproof_evidence,
                            failure_reason="FREEZE_EPOCH_CHANGED",
                            p5_subresult=observation.p5_subresult,
                        )
                        unreconciled_postproof = postproof_evidence
                    else:
                        final_evidence = {
                            **dict(observation.evidence),
                            "final_postproof": {
                                "generation_matches": generation_matches,
                                "ref_matches": ref_matches,
                                "expected_post_state": expected_state,
                                "observed_ref": getattr(postproof, "ref", None),
                                "observed_object_type": getattr(postproof, "ref_object_type", None),
                                "observed_sha": getattr(postproof, "ref_sha", None),
                                "post": postproof_record,
                            },
                        }
                        observation = CaseObservation(
                            observation.status, final_evidence,
                            provider_status=observation.provider_status,
                            failure_reason=observation.failure_reason,
                            p5_subresult=observation.p5_subresult,
                        )
            evidence = dict(observation.evidence)
            evidence.setdefault("request_sha256", request.request_sha256)
            evidence.setdefault("plan_sha256", request.plan_sha256)
            evidence.setdefault("approval_receipt_sha256", approval.receipt_sha256)
            evidence.setdefault("failure_reason", observation.failure_reason)
            result = _result(
                request, case_id, observation.status, evidence,
                p5_subresult=observation.p5_subresult,
            )
            try:
                journal.append_case_result(result, evidence)
            except Exception:
                return _result(
                    request, case_id, "BLOCKED_UNKNOWN",
                    {"reason": "CASE_RESULT_WRITE_UNKNOWN", "prior_evidence_sha256": result["evidence_sha256"]},
                    p5_subresult=observation.p5_subresult,
                )
            if unreconciled_postproof is not None:
                retain = getattr(adapter, "retain_unreconciled_case_postproof", None)
                if callable(retain):
                    retain(request, case_id, unreconciled_postproof)
            if result["unresolved"] or case_id == "4g":
                return result
    raise RehearsalError("DRIVER_SEQUENCE_TERMINAL_STATE_INVALID")


__all__ = [
    "APPROVAL_MARKER_PREFIX", "APPROVAL_MARKER_SCHEMA", "CASE_ORDER",
    "CAPABILITY_4D_SCHEMA", "DRIVER_RETURN_SCHEMA", "OPERATION_TABLE",
    "OWNER_SEED_CLAIM_SCHEMA", "REQUEST_SCHEMA", "REQUEST_SCHEMA_V3",
    "SEED_ACTION_JOURNAL_SCHEMA", "SEED_BUNDLE_SCHEMA",
    "SEED_FAILURE_RECEIPT_SCHEMA", "SeedReceiptEvidence", "SeedReceiptError",
    "STATUS_AWARE_VECTORS",
    "AnonymousGitHubTransport", "CaseObservation", "RehearsalError",
    "ValidatedRouteRequest", "VerifiedRunApproval", "run_rehearsal",
    "CANONICAL_OPERATION_TABLE_SHA256", "rehydrate_approval_envelope",
    "serialize_approval_envelope",
    "serialize_seed_receipt_evidence", "serialize_seed_receipt_evidence_pair",
    "rehydrate_seed_receipt_evidence", "rehydrate_seed_receipt_evidence_pair",
    "validate_approval_receipt", "validate_driver_result",
    "validate_rehearsal_route_trigger", "validate_route_request_bytes",
    "validate_seed_receipt_bundle",
    "verify_run_approval",
]
