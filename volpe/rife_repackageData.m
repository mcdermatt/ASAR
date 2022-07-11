%repackage data

inspvax.nx1465 = nx1465;
inspvax.msgln1465 = msgln1465;
inspvax.wk1465 = wk1465;
inspvax.ms1465 = ms1465;
inspvax.sow1465 = sow1465;
inspvax.insstat = insstat;
inspvax.inslat = inslat;
inspvax.inslon = inslon;
inspvax.inshgt = inshgt;
inspvax.insght = insght;
inspvax.insnorthvel = insnorthvel;
inspvax.inseastvel = inseastvel;
inspvax.insupvel = insupvel;
inspvax.insroll = insroll;
inspvax.inspitch = inspitch;
inspvax.insazim = insazim;
inspvax.insrollstd = insrollstd;
inspvax.inspitchstd = inspitchstd;
inspvax.insazimstd = insazimstd;


rawimusx.nx1462 = nx1462;
rawimusx.msgln1462 = msgln1462;
rawimusx.wk1462 = wk1462;
rawimusx.ms1462 = ms1462;
rawimusx.sow1462 = sow1462;
rawimusx.rawimuinfo = rawimuinfo;
rawimusx.rawimutype =rawimutype;
rawimusx.rawimugnsswk = rawimugnsswk;
rawimusx.rawimugnsssow = rawimugnsssow;
rawimusx.rawimustatus = rawimustatus;
rawimusx.rawimuzaccel = rawimuzaccel;
rawimusx.rawimuyaccel = rawimuyaccel;
rawimusx.rawimuxaccel = rawimuxaccel;
rawimusx.rawimuzgyro = rawimuzgyro;
rawimusx.rawimuygyro = rawimuygyro;
rawimusx.rawimuxgyro = rawimuxgyro;


psrpos.nx47 = nx47;
psrpos.msgln47 = msgln47;
psrpos.wk47 = wk47;
psrpos.ms47 = ms47;
psrpos.sow47 = sow47;
psrpos.psrsolstat = psrsolstat;
psrpos.psrpostype = psrpostype;
psrpos.psrlat = psrlat;
psrpos.psrlon = psrlon;
psrpos.psrhgt = psrhgt;
psrpos.psrundulation = psrundulation;
psrpos.psrdatum_id = psrdatum_id;
psrpos.psrlatstd = psrlatstd;
psrpos.psrlonstd = psrlonstd;
psrpos.psrhgtstd = psrhgtstd;

ppppos.nx1538 = nx1538;
ppppos.msgln1538 = msgln1538;
ppppos.wk1538 = wk1538;
ppppos.ms1538 = ms1538;
ppppos.sow1538 = sow1538;
ppppos.pppstatus = pppstatus;
ppppos.ppppostype = ppppostype;
ppppos.ppplat = ppplat;
ppppos.ppplon = ppplon;
ppppos.ppphgt = ppphgt;
ppppos.pppundulation = pppundulation;
ppppos.pppdatum_id = pppdatum_id;
ppppos.ppplatstd = ppplatstd;
ppppos.ppplonstd = ppplonstd;
ppppos.ppphgtstd = ppphgtstd;
    
bestgnssvel.nx1430 = nx1430;
bestgnssvel.msgln1430 = msgln1430;
bestgnssvel.wk1430 = wk1430;
bestgnssvel.ms1430 = ms1430;
bestgnssvel.sow1430 = sow1430;
bestgnssvel.gnssvelsolstat = gnssvelsolstat;
bestgnssvel.gnssveltype = gnssveltype;
bestgnssvel.gnssvellatency = gnssvellatency;
bestgnssvel.gnssvelage = gnssvelage;
bestgnssvel.gnssvelhorspd = gnssvelhorspd;
bestgnssvel.gnssveltrkgnd = gnssveltrkgnd;
bestgnssvel.gnssvelvertspd = gnssvelvertspd;

insconfig.nx1945 = nx1945;
insconfig.msgln1945 = msgln1945;
insconfig.wk1945 = wk1945;
insconfig.ms1945 = ms1945;
insconfig.sow1945 =sow1945;
insconfig.imutype = imutype;
insconfig.initalignvel = initalignvel;
insconfig.imuprofile = imuprofile;
insconfig.insupdates = insupdates;
insconfig.alignmentmode = alignmentmode;
insconfig.ntrans = ntrans;
insconfig.transvecs = transvecs;
insconfig.nrots = nrots;
insconfig.rotvecs = rotvecs;

%save signageData inspvax rawimusx psrpos ppppos bestgnssvel insconfig
% save('roundaboutData.mat', bestgnssvel, insconfig, inspvax, ppppos, psrpos, rawimusx)
