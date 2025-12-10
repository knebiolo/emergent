# sockeye.py except Exception inventory

Generated: 20251209T145518Z

Total matches: 344

## Line 80
```
   77:             logger.exception('%s | %s | %s', msg, exc, ctx_s)
   78:         else:
   79:             logger.exception('%s | %s', msg, exc)
   80:     except Exception:
   81:         try:
   82:             import traceback
   83:             print('LOGGING FAILURE:', msg, exc)
```

## Line 85
```
   82:             import traceback
   83:             print('LOGGING FAILURE:', msg, exc)
   84:             print(traceback.format_exc())
   85:         except Exception:
   86:             # avoid further recursion; give up silently
   87:             pass
   88: 
```

## Line 108
```
  105:     except (ValueError, TypeError, IndexError, AttributeError):
  106:         logger.exception('%s: failed to build cKDTree for provided points', name)
  107:         return None
  108:     except Exception:
  109:         logger.exception('%s: unexpected error while building cKDTree; re-raising', name)
  110:         raise
  111: 
```

## Line 128
```
  125:     _HAS_NUMBA = False
  126:     try:
  127:         logger.warning('Numba import failed; falling back to pure-Python implementations: %s', e)
  128:     except Exception as _log_e:
  129:         try:
  130:             logger.exception('Failed while logging numba import warning: %s', _log_e)
  131:         except Exception as e:
```

## Line 131
```
  128:     except Exception as _log_e:
  129:         try:
  130:             logger.exception('Failed while logging numba import warning: %s', _log_e)
  131:         except Exception as e:
  132:             try:
  133:                 logger.exception('Error computing schooling loop step for agent %s: %s', i, e)
  134:             except Exception:
```

## Line 134
```
  131:         except Exception as e:
  132:             try:
  133:                 logger.exception('Error computing schooling loop step for agent %s: %s', i, e)
  134:             except Exception:
  135:                 try:
  136:                     print('Logging failure in schooling loop handler:', e)
  137:                 except Exception as e:
```

## Line 137
```
  134:             except Exception:
  135:                 try:
  136:                     print('Logging failure in schooling loop handler:', e)
  137:                 except Exception as e:
  138:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=136)
  139:                     pass
  140: except Exception as e:
```

## Line 140
```
  137:                 except Exception as e:
  138:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=136)
  139:                     pass
  140: except Exception as e:
  141:     # Log and re-raise unexpected exceptions during import
  142:     logger.exception('Unexpected error while importing numba: %s', e)
  143:     raise
```

## Line 204
```
  201:                 out[k] = getattr(self, k)
  202:             except AttributeError:
  203:                 out[k] = None
  204:             except Exception as e:
  205:                 _safe_log_exception('Unexpected error exporting weight; setting to None', e, key=k)
  206:                 out[k] = None
  207:         return out
```

## Line 225
```
  222:         except (OSError, IOError) as e:
  223:             try:
  224:                 logger.exception('Failed to log save operation for behavioral weights: %s', e)
  225:             except Exception as _log_e:
  226:                 try:
  227:                     # final best-effort logging
  228:                     print('Logging failure in BehavioralWeights.save:', _log_e)
```

## Line 229
```
  226:                 try:
  227:                     # final best-effort logging
  228:                     print('Logging failure in BehavioralWeights.save:', _log_e)
  229:                 except Exception as e:
  230:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=227)
  231:                     pass
  232:     
```

## Line 244
```
  241:         except (OSError, IOError, ValueError) as e:
  242:             try:
  243:                 logger.exception('Failed to log load operation for behavioral weights: %s', e)
  244:             except Exception as _log_e:
  245:                 try:
  246:                     print('Logging failure in BehavioralWeights.load:', _log_e)
  247:                 except Exception as e:
```

## Line 247
```
  244:             except Exception as _log_e:
  245:                 try:
  246:                     print('Logging failure in BehavioralWeights.load:', _log_e)
  247:                 except Exception as e:
  248:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=244)
  249:                     pass
  250:         except Exception as e:
```

## Line 250
```
  247:                 except Exception as e:
  248:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=244)
  249:                     pass
  250:         except Exception as e:
  251:             try:
  252:                 logger.exception('Unexpected error while logging load operation; re-raising: %s', e)
  253:             except Exception as _log_e:
```

## Line 253
```
  250:         except Exception as e:
  251:             try:
  252:                 logger.exception('Unexpected error while logging load operation; re-raising: %s', e)
  253:             except Exception as _log_e:
  254:                 try:
  255:                     print('Logging failure while handling exception in BehavioralWeights.load:', _log_e)
  256:                 except Exception as e:
```

## Line 256
```
  253:             except Exception as _log_e:
  254:                 try:
  255:                     print('Logging failure while handling exception in BehavioralWeights.load:', _log_e)
  256:                 except Exception as e:
  257:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=252)
  258:                     pass
  259:             raise
```

## Line 675
```
  672:             except (ValueError, IndexError, TypeError, AttributeError):
  673:                 logger.exception('Collision metric computation failed; returning 0 collisions')
  674:                 metrics['collision_count'] = 0
  675:             except Exception as e:
  676:                 logger.exception('Unexpected error computing collision metrics; re-raising: %s', e)
  677:                 raise
  678: 
```

## Line 702
```
  699:             except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
  700:                 try:
  701:                     logger.exception('Error computing dry/shallow counts; defaulting to 0: %s', e)
  702:                 except Exception as _log_e:
  703:                     try:
  704:                         logger.exception('Failed while logging dry/shallow counts error: %s', _log_e)
  705:                     except Exception as e:
```

## Line 705
```
  702:                 except Exception as _log_e:
  703:                     try:
  704:                         logger.exception('Failed while logging dry/shallow counts error: %s', _log_e)
  705:                     except Exception as e:
  706:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=700)
  707:                         pass
  708:                 metrics['dry_count'] = 0
```

## Line 710
```
  707:                         pass
  708:                 metrics['dry_count'] = 0
  709:                 metrics['shallow_count'] = 0
  710:             except Exception as e:
  711:                 logger.exception('Unexpected error computing dry/shallow counts; re-raising: %s', e)
  712:                 raise
  713:         
```

## Line 772
```
  769:             except (ValueError, TypeError, IndexError, AttributeError) as e:
  770:                 try:
  771:                     logger.exception('Error computing mean upstream velocity; defaulting to 0.0: %s', e)
  772:                 except Exception as e:
  773:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=772)
  774:                     pass
  775:                 metrics['mean_upstream_velocity'] = 0.0
```

## Line 776
```
  773:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=772)
  774:                     pass
  775:                 metrics['mean_upstream_velocity'] = 0.0
  776:             except Exception as e:
  777:                 logger.exception('Unexpected error computing mean upstream velocity; re-raising: %s', e)
  778:                 raise
  779:             self.sim._prev_centerline_meas = self.sim.centerline_meas.copy()
```

## Line 823
```
  820:         except (ValueError, TypeError, AttributeError) as e:
  821:             try:
  822:                 logger.exception('logger.info failed during RL training start: %s', e)
  823:             except Exception as e:
  824:                 try:
  825:                     logger.exception('Unexpected error in schooling metric aggregation: %s', e)
  826:                 except Exception:
```

## Line 826
```
  823:             except Exception as e:
  824:                 try:
  825:                     logger.exception('Unexpected error in schooling metric aggregation: %s', e)
  826:                 except Exception:
  827:                     try:
  828:                         print('Logging failure in schooling metric aggregation:', e)
  829:                     except Exception as e:
```

## Line 829
```
  826:                 except Exception:
  827:                     try:
  828:                         print('Logging failure in schooling metric aggregation:', e)
  829:                     except Exception as e:
  830:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=828)
  831:                         pass
  832:         
```

## Line 853
```
  850:                 except (ValueError, TypeError, AttributeError) as e:
  851:                     try:
  852:                         logger.exception('logger.info failed during episode logging: %s', e)
  853:                     except Exception as e:
  854:                         try:
  855:                             logger.exception('Failed during centroid computation: %s', e)
  856:                         except Exception:
```

## Line 856
```
  853:                     except Exception as e:
  854:                         try:
  855:                             logger.exception('Failed during centroid computation: %s', e)
  856:                         except Exception:
  857:                             try:
  858:                                 print('Logging failure in centroid computation:', e)
  859:                             except Exception as e:
```

## Line 859
```
  856:                         except Exception:
  857:                             try:
  858:                                 print('Logging failure in centroid computation:', e)
  859:                             except Exception as e:
  860:                                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=857)
  861:                                 pass
  862:                 
```

## Line 885
```
  882:                     except (AttributeError, NameError, RuntimeError) as e:
  883:                         try:
  884:                             logger.exception('Failed to delete _last_drag_reductions during cleanup: %s', e)
  885:                         except Exception as e:
  886:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=882)
  887:                             pass
  888:                 try:
```

## Line 893
```
  890:                 except (RuntimeError, OSError) as e:
  891:                     try:
  892:                         logger.exception('gc.collect() raised runtime error during cleanup: %s', e)
  893:                     except Exception as e:
  894:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=889)
  895:                         pass
  896:             except (ImportError, RuntimeError, OSError) as e:
```

## Line 899
```
  896:             except (ImportError, RuntimeError, OSError) as e:
  897:                 try:
  898:                     logger.exception('GC/cleanup block encountered runtime error: %s', e)
  899:                 except Exception as e:
  900:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=894)
  901:                     pass
  902:             except Exception:
```

## Line 902
```
  899:                 except Exception as e:
  900:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=894)
  901:                     pass
  902:             except Exception:
  903:                 logger.exception('Unexpected error during GC/cleanup; re-raising')
  904:                 raise
  905:         
```

## Line 914
```
  911:             except (ValueError, TypeError, AttributeError) as e:
  912:                 try:
  913:                     logger.exception('logger.info failed when finishing training: %s', e)
  914:                 except Exception as e:
  915:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=914)
  916:                     pass
  917:         
```

## Line 952
```
  949:                 if name_pattern in p or name_pattern in obj.name.lower():
  950:                     try:
  951:                         shape = obj.shape
  952:                     except Exception:
  953:                         shape = None
  954:                     candidates.append((path, shape))
  955: 
```

## Line 1034
```
 1031:         if self.tree is None:
 1032:             try:
 1033:                 logger.warning('HECRAS plan: KDTree build failed; certain queries will be disabled')
 1034:             except Exception as e:
 1035:                 try:
 1036:                     logger.exception('Error while updating alignment scores: %s', e)
 1037:                 except Exception:
```

## Line 1037
```
 1034:             except Exception as e:
 1035:                 try:
 1036:                     logger.exception('Error while updating alignment scores: %s', e)
 1037:                 except Exception:
 1038:                     try:
 1039:                         print('Logging failure in alignment update:', e)
 1040:                     except Exception as e:
```

## Line 1040
```
 1037:                 except Exception:
 1038:                     try:
 1039:                         print('Logging failure in alignment update:', e)
 1040:                     except Exception as e:
 1041:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1039)
 1042:                         pass
 1043: 
```

## Line 1076
```
 1073:     """
 1074:     try:
 1075:         from emergent.salmon_abm.hecras_helpers import infer_wetted_perimeter_from_hecras as _central
 1076:     except Exception as e:
 1077:         logger.exception('Failed to import central hecras helper; falling back to local import: %s', e)
 1078:         # local fallback: import from package-relative path
 1079:         from .hecras_helpers import infer_wetted_perimeter_from_hecras as _central
```

## Line 1087
```
 1084:     if timestep != 0 and verbose:
 1085:         try:
 1086:             logger.warning("[consolidation] central HECRAS helper ignores 'timestep' argument (requested %s); using first timestep", timestep)
 1087:         except Exception as _log_e:
 1088:             try:
 1089:                 logger.exception("Failed while logging HECRAS timestep warning: %s", _log_e)
 1090:             except Exception as e:
```

## Line 1090
```
 1087:         except Exception as _log_e:
 1088:             try:
 1089:                 logger.exception("Failed while logging HECRAS timestep warning: %s", _log_e)
 1090:             except Exception as e:
 1091:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1088)
 1092:                 pass
 1093: 
```

## Line 1100
```
 1097: def compute_distance_to_bank_hecras(wetted_info, coords, median_spacing=None):
 1098:     try:
 1099:         from emergent.salmon_abm.hecras_helpers import compute_distance_to_bank_hecras as _central
 1100:     except Exception as e:
 1101:         logger.exception('Failed to import central compute_distance_to_bank_hecras; falling back to local import: %s', e)
 1102:         from .hecras_helpers import compute_distance_to_bank_hecras as _central
 1103:     return _central(wetted_info, coords, median_spacing=median_spacing)
```

## Line 1110
```
 1107:                                            min_distance_threshold=None, min_length=50):
 1108:     try:
 1109:         from emergent.salmon_abm.hecras_helpers import derive_centerline_from_hecras_distance as _central
 1110:     except Exception as e:
 1111:         logger.exception('Failed to import central derive_centerline_from_hecras_distance; falling back to local import: %s', e)
 1112:         from .hecras_helpers import derive_centerline_from_hecras_distance as _central
 1113:     return _central(coords, distances, wetted_mask, crs=crs, min_distance_threshold=min_distance_threshold, min_length=min_length)
```

## Line 1154
```
 1151:     if len(valid_coords) == 0:
 1152:         try:
 1153:             logger.warning("No valid wetted cells for centerline extraction")
 1154:         except Exception as e:
 1155:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1151)
 1156:             pass
 1157:         return None
```

## Line 1169
```
 1166:     
 1167:     try:
 1168:         logger.info("Ridge cells (distance >= %0.2fm): %,d", min_distance_threshold, len(ridge_coords))
 1169:     except Exception as e:
 1170:         try:
 1171:             logger.exception('Error during sksurv import fallback check: %s', e)
 1172:         except Exception as e:
```

## Line 1172
```
 1169:     except Exception as e:
 1170:         try:
 1171:             logger.exception('Error during sksurv import fallback check: %s', e)
 1172:         except Exception as e:
 1173:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1168)
 1174:             pass
 1175:     
```

## Line 1179
```
 1176:     if len(ridge_coords) < 10:
 1177:         try:
 1178:             logger.warning("Too few ridge cells for centerline extraction")
 1179:         except Exception as e:
 1180:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1174)
 1181:             pass
 1182:         return None
```

## Line 1195
```
 1192:     if ridge_tree is None:
 1193:         try:
 1194:             logger.warning('ridge_tree could not be built; aborting ordered ridge extraction')
 1195:         except Exception as e:
 1196:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1195)
 1197:             pass
 1198:         return None
```

## Line 1233
```
 1230:     
 1231:     try:
 1232:         logger.info("Centerline extracted: %0.2fm long", centerline.length)
 1233:     except Exception as e:
 1234:         try:
 1235:             logger.exception('Unexpected error during sksurv import check: %s', e)
 1236:         except Exception as e:
```

## Line 1236
```
 1233:     except Exception as e:
 1234:         try:
 1235:             logger.exception('Unexpected error during sksurv import check: %s', e)
 1236:         except Exception as e:
 1237:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1235)
 1238:             pass
 1239:     
```

## Line 1243
```
 1240:     if centerline.length < min_length:
 1241:         try:
 1242:             logger.warning("Centerline too short (%0.2fm < %sm)", centerline.length, min_length)
 1243:         except Exception as e:
 1244:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1241)
 1245:             pass
 1246:         return None
```

## Line 1255
```
 1252:     """Wrapper to centralized fast centerline helper."""
 1253:     try:
 1254:         from emergent.salmon_abm.hecras_helpers import extract_centerline_fast_hecras as _central
 1255:     except Exception:
 1256:         from .hecras_helpers import extract_centerline_fast_hecras as _central
 1257:     return _central(plan_path, depth_threshold=depth_threshold, sample_fraction=sample_fraction, min_length=min_length)
 1258: 
```

## Line 1299
```
 1296:         logger.info("%s", "="*80)
 1297:         logger.info("HECRAS GEOMETRY INITIALIZATION")
 1298:         logger.info("%s", "="*80)
 1299:     except Exception as e:
 1300:         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1296)
 1301:         pass
 1302:     
```

## Line 1306
```
 1303:     # Step 1: Load HECRAS geometry and build KDTree
 1304:     try:
 1305:         logger.info("1. Loading HECRAS plan and building KDTree...")
 1306:     except Exception as e:
 1307:         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1302)
 1308:         pass
 1309:     with h5py.File(plan_path, 'r') as hdf:
```

## Line 1322
```
 1319:     hecras_map = simulation._hecras_maps[key]
 1320:     try:
 1321:         logger.info("Loaded %d HECRAS cells", len(coords))
 1322:     except Exception as e:
 1323:         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1317)
 1324:         pass
 1325:     
```

## Line 1329
```
 1326:     # Step 2: Fast centerline extraction
 1327:     try:
 1328:         logger.info("2. Extracting centerline...")
 1329:     except Exception as e:
 1330:         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1329)
 1331:         pass
 1332:     centerline = extract_centerline_fast_hecras(
```

## Line 1342
```
 1339:     if centerline is None:
 1340:         try:
 1341:             logger.warning("Failed to extract centerline!")
 1342:         except Exception as e:
 1343:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1341)
 1344:             pass
 1345: 
```

## Line 1352
```
 1349:         if hecras_verbose:
 1350:             try:
 1351:                 logger.info("2b. Inferring wetted perimeter (vectorizing)...")
 1352:             except Exception as e:
 1353:                 try:
 1354:                     logger.exception('Failed while attempting to warn about logger.info in BehavioralWeights.save: %s', e)
 1355:                 except Exception as e:
```

## Line 1355
```
 1352:             except Exception as e:
 1353:                 try:
 1354:                     logger.exception('Failed while attempting to warn about logger.info in BehavioralWeights.save: %s', e)
 1355:                 except Exception as e:
 1356:                     try:
 1357:                         logger.exception('Failed to finalize cohesion calculation: %s', e)
 1358:                     except Exception:
```

## Line 1358
```
 1355:                 except Exception as e:
 1356:                     try:
 1357:                         logger.exception('Failed to finalize cohesion calculation: %s', e)
 1358:                     except Exception:
 1359:                         try:
 1360:                             print('Logging failure in cohesion finalization:', e)
 1361:                         except Exception as e:
```

## Line 1361
```
 1358:                     except Exception:
 1359:                         try:
 1360:                             print('Logging failure in cohesion finalization:', e)
 1361:                         except Exception as e:
 1362:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1359)
 1363:                             pass
 1364:         wetted_info = infer_wetted_perimeter_from_hecras(plan_path, depth_threshold=depth_threshold, timestep=0, verbose=False)
```

## Line 1371
```
 1368:         if hecras_verbose:
 1369:             try:
 1370:                 logger.info("Perimeter points: %d", 0 if perimeter_points is None else len(perimeter_points))
 1371:             except Exception as e:
 1372:                 try:
 1373:                     logger.exception('Failed while attempting to warn about logger.info in BehavioralWeights.load: %s', e)
 1374:                 except Exception as e:
```

## Line 1374
```
 1371:             except Exception as e:
 1372:                 try:
 1373:                     logger.exception('Failed while attempting to warn about logger.info in BehavioralWeights.load: %s', e)
 1374:                 except Exception as e:
 1375:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1371)
 1376:                     pass
 1377:     except Exception:
```

## Line 1377
```
 1374:                 except Exception as e:
 1375:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1371)
 1376:                     pass
 1377:     except Exception:
 1378:         perimeter_points = None
 1379:         perimeter_cells = None
 1380:         median_spacing = None
```

## Line 1387
```
 1384:     if create_rasters:
 1385:         try:
 1386:             logger.info("3. Creating regular grid rasters...")
 1387:         except Exception as e:
 1388:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1383)
 1389:             pass
 1390:         
```

## Line 1404
```
 1401:         
 1402:         try:
 1403:             logger.info("Grid dimensions: %d x %d at %0.2fm resolution", height, width, cell_size)
 1404:         except Exception as e:
 1405:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1399)
 1406:             pass
 1407:         
```

## Line 1414
```
 1411:         # Map initial HECRAS fields to rasters
 1412:         try:
 1413:             logger.info("4. Mapping HECRAS fields to rasters...")
 1414:         except Exception as e:
 1415:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1414)
 1416:             pass
 1417:         map_hecras_to_env_rasters(simulation, plan_path, field_names=fields, k=1)  # k=1 for speed
```

## Line 1452
```
 1449:     except ImportError:
 1450:         try:
 1451:             from .hecras_helpers import map_hecras_for_agents as _central
 1452:         except Exception:
 1453:             try:
 1454:                 logger.exception('Failed to import hecras_helpers.map_hecras_for_agents')
 1455:             except Exception as e:
```

## Line 1455
```
 1452:         except Exception:
 1453:             try:
 1454:                 logger.exception('Failed to import hecras_helpers.map_hecras_for_agents')
 1455:             except Exception as e:
 1456:                 try:
 1457:                     logger.exception('Failed while logging collision metric computation failure: %s', e)
 1458:                 except Exception as e:
```

## Line 1458
```
 1455:             except Exception as e:
 1456:                 try:
 1457:                     logger.exception('Failed while logging collision metric computation failure: %s', e)
 1458:                 except Exception as e:
 1459:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1457)
 1460:                     pass
 1461:             raise
```

## Line 1473
```
 1470:     except ImportError:
 1471:         try:
 1472:             from .hecras_helpers import ensure_hdf_coords_from_hecras as _central
 1473:         except Exception:
 1474:             try:
 1475:                 logger.exception('Failed to import local hecras_helpers.ensure_hdf_coords_from_hecras')
 1476:             except Exception as e:
```

## Line 1476
```
 1473:         except Exception:
 1474:             try:
 1475:                 logger.exception('Failed to import local hecras_helpers.ensure_hdf_coords_from_hecras')
 1476:             except Exception as e:
 1477:                 try:
 1478:                     logger.exception('Failed while logging dry/shallow counts computation error: %s', e)
 1479:                 except Exception as e:
```

## Line 1479
```
 1476:             except Exception as e:
 1477:                 try:
 1478:                     logger.exception('Failed while logging dry/shallow counts computation error: %s', e)
 1479:                 except Exception as e:
 1480:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1477)
 1481:                     pass
 1482:             raise
```

## Line 1505
```
 1502:         existing = np.asarray(dset_x[:])
 1503:         # Check if uninit: all non-finite OR all zeros
 1504:         needs_populate = not np.isfinite(existing).any() or np.allclose(existing, 0.0)
 1505:     except Exception:
 1506:         needs_populate = True
 1507: 
 1508:     if needs_populate:
```

## Line 1528
```
 1525:     """Wrapper: delegate to `hecras_helpers.map_hecras_to_env_rasters` with timestep."""
 1526:     try:
 1527:         from emergent.salmon_abm.hecras_helpers import map_hecras_to_env_rasters as _central
 1528:     except Exception:
 1529:         from .hecras_helpers import map_hecras_to_env_rasters as _central
 1530:     return _central(simulation, plan_path, field_names=field_names, k=k, timestep=timestep)
 1531: #                             hw.flush()
```

## Line 1532
```
 1529:         from .hecras_helpers import map_hecras_to_env_rasters as _central
 1530:     return _central(simulation, plan_path, field_names=field_names, k=k, timestep=timestep)
 1531: #                             hw.flush()
 1532: #                         except Exception:
 1533: #                             pass
 1534: #                 except Exception:
 1535: #                     pass
```

## Line 1534
```
 1531: #                             hw.flush()
 1532: #                         except Exception:
 1533: #                             pass
 1534: #                 except Exception:
 1535: #                     pass
 1536: #     except Exception:
 1537: #         pass
```

## Line 1536
```
 1533: #                             pass
 1534: #                 except Exception:
 1535: #                     pass
 1536: #     except Exception:
 1537: #         pass
 1538: 
 1539: # # End HECRAS helpers
```

## Line 1571
```
 1568:     except (RuntimeError, ValueError, TypeError, IndexError, AttributeError) as e:
 1569:         try:
 1570:             logger.exception('KDTree spacing estimation failed; using fallback median spacing: %s', e)
 1571:         except Exception as e:
 1572:             try:
 1573:                 logger.exception('Failed while logging HECRAS geometry init message: %s', e)
 1574:             except Exception as e:
```

## Line 1574
```
 1571:         except Exception as e:
 1572:             try:
 1573:                 logger.exception('Failed while logging HECRAS geometry init message: %s', e)
 1574:             except Exception as e:
 1575:                 try:
 1576:                     logger.exception('BehavioralWeights.to_dict: unexpected error while getting attribute %s: %s', k, e)
 1577:                 except Exception as e:
```

## Line 1577
```
 1574:             except Exception as e:
 1575:                 try:
 1576:                     logger.exception('BehavioralWeights.to_dict: unexpected error while getting attribute %s: %s', k, e)
 1577:                 except Exception as e:
 1578:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1574)
 1579:                     pass
 1580:         # fallback: estimate spacing using bounding box / sqrt(n)
```

## Line 1584
```
 1581:         bbox = coords.max(axis=0) - coords.min(axis=0)
 1582:         approx_cell = float(np.sqrt((bbox[0] * bbox[1]) / max(1, n)))
 1583:         median_spacing = approx_cell
 1584:     except Exception:
 1585:         logger.exception('Unexpected error during spacing KDTree; re-raising')
 1586:         raise
 1587:     # second column are nearest neighbor distances
```

## Line 1606
```
 1603:     try:
 1604:         from rasterio.transform import Affine as _Affine
 1605:         AffineLocal = _Affine
 1606:     except Exception:
 1607:         try:
 1608:             from affine import Affine as _Affine
 1609:             AffineLocal = _Affine
```

## Line 1610
```
 1607:         try:
 1608:             from affine import Affine as _Affine
 1609:             AffineLocal = _Affine
 1610:         except Exception:
 1611:             # last resort: build a simple stand-in
 1612:             class AffineLocal:
 1613:                 def __init__(self, a, b, c, d, e, f):
```

## Line 1707
```
 1704:         try:
 1705:             import cupy as cp
 1706:             return cp
 1707:         except Exception:
 1708:             try:
 1709:                 logger.info("CuPy not found. Falling back to Numpy.")
 1710:             except Exception as e:
```

## Line 1710
```
 1707:         except Exception:
 1708:             try:
 1709:                 logger.info("CuPy not found. Falling back to Numpy.")
 1710:             except Exception as e:
 1711:                 try:
 1712:                     logger.exception('BehavioralWeights.from_dict encountered unexpected error: %s', e)
 1713:                 except Exception as e:
```

## Line 1713
```
 1710:             except Exception as e:
 1711:                 try:
 1712:                     logger.exception('BehavioralWeights.from_dict encountered unexpected error: %s', e)
 1713:                 except Exception as e:
 1714:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1709)
 1715:                     pass
 1716:             import numpy as np
```

## Line 1751
```
 1748:         rows = np.rint(rows).astype(int)
 1749:         cols = np.rint(cols).astype(int)
 1750:         return rows, cols
 1751:     except Exception:
 1752:         # Fallback to per-point multiplication
 1753:         inv_transform = ~transform
 1754:         pixels = [inv_transform * (x, y) for x, y in zip(X, Y)]
```

## Line 1779
```
 1776:     """
 1777:     try:
 1778:         key = id(transform)
 1779:     except Exception:
 1780:         return ~transform
 1781:     cache = getattr(sim, '_inv_transform_cache', None)
 1782:     if cache is None:
```

## Line 1789
```
 1786:     if inv is None:
 1787:         try:
 1788:             inv = ~transform
 1789:         except Exception:
 1790:             # best-effort: return direct invertible object
 1791:             return ~transform
 1792:         cache[key] = inv
```

## Line 1811
```
 1808:             except (OSError, IOError) as e:
 1809:                 try:
 1810:                     logger.exception('hdf.flush() failed (runtime): %s', e)
 1811:                 except Exception as e:
 1812:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1806)
 1813:                     pass
 1814:             except Exception:
```

## Line 1814
```
 1811:                 except Exception as e:
 1812:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1806)
 1813:                     pass
 1814:             except Exception:
 1815:                 logger.exception('Unexpected error while calling hdf.flush(); re-raising')
 1816:                 raise
 1817:         # h5py Group has .file attribute referencing the File object
```

## Line 1826
```
 1823:             except (OSError, IOError) as e:
 1824:                 try:
 1825:                     logger.exception('hdf.file.flush() failed (runtime): %s', e)
 1826:                 except Exception as e:
 1827:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1826)
 1828:                     pass
 1829:             except Exception:
```

## Line 1829
```
 1826:                 except Exception as e:
 1827:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1826)
 1828:                     pass
 1829:             except Exception:
 1830:                 logger.exception('Unexpected error while calling hdf.file.flush(); re-raising')
 1831:                 raise
 1832:         # fallback: try to open by filename and flush
```

## Line 1842
```
 1839:                     except (OSError, IOError) as e:
 1840:                         try:
 1841:                             logger.exception('h5py.File(%s).flush() failed (runtime): %s', fname, e)
 1842:                         except Exception as e:
 1843:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1841)
 1844:                             pass
 1845:                     except Exception:
```

## Line 1845
```
 1842:                         except Exception as e:
 1843:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1841)
 1844:                             pass
 1845:                     except Exception:
 1846:                         logger.exception('Unexpected error while flushing reopened HDF file; re-raising')
 1847:                         raise
 1848:             except (OSError, IOError) as e:
```

## Line 1851
```
 1848:             except (OSError, IOError) as e:
 1849:                 try:
 1850:                     logger.exception('Failed to reopen HDF file %s for flush (runtime): %s', fname, e)
 1851:                 except Exception as e:
 1852:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1849)
 1853:                     pass
 1854:             except Exception:
```

## Line 1854
```
 1851:                 except Exception as e:
 1852:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1849)
 1853:                     pass
 1854:             except Exception:
 1855:                 logger.exception('Unexpected error while reopening HDF file %s for flush; re-raising', fname)
 1856:                 raise
 1857:     except Exception:
```

## Line 1857
```
 1854:             except Exception:
 1855:                 logger.exception('Unexpected error while reopening HDF file %s for flush; re-raising', fname)
 1856:                 raise
 1857:     except Exception:
 1858:         try:
 1859:             logger.exception('safe_flush encountered unexpected error; re-raising')
 1860:         except Exception as e:
```

## Line 1860
```
 1857:     except Exception:
 1858:         try:
 1859:             logger.exception('safe_flush encountered unexpected error; re-raising')
 1860:         except Exception as e:
 1861:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=1857)
 1862:             pass
 1863:         raise
```

## Line 1871
```
 1868:     from numba import njit, prange
 1869:     import math
 1870:     _HAS_NUMBA = True
 1871: except Exception:
 1872:     _HAS_NUMBA = False
 1873: 
 1874: 
```

## Line 2355
```
 2352:     except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
 2353:         try:
 2354:             logger.exception('Numba precompile at import time failed (expected runtime issue); continuing without warmed numba kernels: %s', e)
 2355:         except Exception as e:
 2356:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2351)
 2357:             pass
 2358:     except Exception:
```

## Line 2358
```
 2355:         except Exception as e:
 2356:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2351)
 2357:             pass
 2358:     except Exception:
 2359:         logger.exception('Numba precompile at import time failed with unexpected error; re-raising')
 2360:         raise
 2361: 
```

## Line 2386
```
 2383:         except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
 2384:             try:
 2385:                 logger.exception('_numba_warmup failed with expected runtime issue; numba kernels may not be available: %s', e)
 2386:             except Exception as e:
 2387:                 try:
 2388:                     logger.exception('BehavioralWeights.save logging failed: %s', e)
 2389:                 except Exception as e:
```

## Line 2389
```
 2386:             except Exception as e:
 2387:                 try:
 2388:                     logger.exception('BehavioralWeights.save logging failed: %s', e)
 2389:                 except Exception as e:
 2390:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2384)
 2391:                     pass
 2392:         except Exception:
```

## Line 2392
```
 2389:                 except Exception as e:
 2390:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2384)
 2391:                     pass
 2392:         except Exception:
 2393:             logger.exception('_numba_warmup failed with unexpected error; re-raising')
 2394:             raise
 2395: 
```

## Line 2402
```
 2399:     except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
 2400:         try:
 2401:             logger.exception('_numba_warmup() failed during module init with expected runtime issue; continuing without warmed kernels: %s', e)
 2402:         except Exception as e:
 2403:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2402)
 2404:             pass
 2405:     except Exception:
```

## Line 2405
```
 2402:         except Exception as e:
 2403:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2402)
 2404:             pass
 2405:     except Exception:
 2406:         logger.exception('_numba_warmup() failed during module init with unexpected error; re-raising')
 2407:         raise
 2408: 
```

## Line 2431
```
 2428:             except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
 2429:                 try:
 2430:                     logger.exception('Exact-shape numba warmup: _compute_drags_numba failed for sim-shaped arrays (runtime issue): %s', e)
 2431:                 except Exception as e:
 2432:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2430)
 2433:                     pass
 2434:             except Exception:
```

## Line 2434
```
 2431:                 except Exception as e:
 2432:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2430)
 2433:                     pass
 2434:             except Exception:
 2435:                 logger.exception('Exact-shape numba warmup: _compute_drags_numba failed with unexpected error; re-raising')
 2436:                 raise
 2437:             try:
```

## Line 2442
```
 2439:             except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
 2440:                 try:
 2441:                     logger.exception('Exact-shape numba warmup: _swim_speeds_numba failed for sim-shaped arrays (runtime issue): %s', e)
 2442:                 except Exception as e:
 2443:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2440)
 2444:                     pass
 2445:             except Exception:
```

## Line 2445
```
 2442:                 except Exception as e:
 2443:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2440)
 2444:                     pass
 2445:             except Exception:
 2446:                 logger.exception('Exact-shape numba warmup: _swim_speeds_numba failed with unexpected error; re-raising')
 2447:                 raise
 2448:             try:
```

## Line 2455
```
 2452:             except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
 2453:                 try:
 2454:                     logger.exception('Exact-shape numba warmup: _assess_fatigue_core failed for sim-shaped arrays (runtime issue): %s', e)
 2455:                 except Exception as e:
 2456:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2452)
 2457:                     pass
 2458:             except Exception:
```

## Line 2458
```
 2455:                 except Exception as e:
 2456:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2452)
 2457:                     pass
 2458:             except Exception:
 2459:                 logger.exception('Exact-shape numba warmup: _assess_fatigue_core failed with unexpected error; re-raising')
 2460:                 raise
 2461:         except Exception:
```

## Line 2461
```
 2458:             except Exception:
 2459:                 logger.exception('Exact-shape numba warmup: _assess_fatigue_core failed with unexpected error; re-raising')
 2460:                 raise
 2461:         except Exception:
 2462:             try:
 2463:                 logger.exception('_numba_warmup_for_sim failed while preparing sim-shaped warmups')
 2464:             except Exception as e:
```

## Line 2464
```
 2461:         except Exception:
 2462:             try:
 2463:                 logger.exception('_numba_warmup_for_sim failed while preparing sim-shaped warmups')
 2464:             except Exception as e:
 2465:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2460)
 2466:                 pass
 2467: 
```

## Line 2718
```
 2715:         if t is None:
 2716:             # try generic transform
 2717:             t = getattr(simulation, 'vel_mag_rast_transform', None)
 2718:     except Exception:
 2719:         t = None
 2720:     if t is None:
 2721:         # fallback to unit pixels
```

## Line 2779
```
 2776:         try:
 2777:             orow, ocol = geo_to_pixel(simulation.depth_rast_transform, [oy], [ox])
 2778:             orow = int(orow[0]); ocol = int(ocol[0])
 2779:         except Exception:
 2780:             orow = None
 2781:         if orow is None or orow < 0 or orow >= h or ocol < 0 or ocol >= w or idx[orow, ocol] < 0:
 2782:             # fallback to nearest wetted cell by Euclidean
```

## Line 2831
```
 2828:         except (OSError, IOError) as e:
 2829:             try:
 2830:                 logger.exception('safe_flush failed after writing along-stream raster (runtime): %s', e)
 2831:             except Exception as e:
 2832:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2826)
 2833:                 pass
 2834:         except Exception:
```

## Line 2834
```
 2831:             except Exception as e:
 2832:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2826)
 2833:                 pass
 2834:         except Exception:
 2835:             logger.exception('Unexpected error during safe_flush after writing along-stream raster; re-raising')
 2836:             raise
 2837:         wrote = True
```

## Line 2854
```
 2851:                     except (OSError, IOError) as e:
 2852:                         try:
 2853:                             logger.exception('hw.flush() failed while reopening HDF (runtime): %s', e)
 2854:                         except Exception as e:
 2855:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2854)
 2856:                             pass
 2857:                     except Exception:
```

## Line 2857
```
 2854:                         except Exception as e:
 2855:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2854)
 2856:                             pass
 2857:                     except Exception:
 2858:                         logger.exception('Unexpected error while flushing reopened HDF; re-raising')
 2859:                         raise
 2860:                     wrote = True
```

## Line 2864
```
 2861:             except (OSError, IOError) as e:
 2862:                 try:
 2863:                     logger.exception('Failed to reopen HDF file %s for write (runtime): %s', fname, e)
 2864:                 except Exception as e:
 2865:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2863)
 2866:                     pass
 2867:                 wrote = False
```

## Line 2868
```
 2865:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=2863)
 2866:                     pass
 2867:                 wrote = False
 2868:             except Exception:
 2869:                 logger.exception('Unexpected error while reopening HDF file %s for write; re-raising', fname)
 2870:                 raise
 2871:         if not wrote:
```

## Line 2943
```
 2940:         t = getattr(simulation, 'depth_rast_transform', None)
 2941:         if t is None:
 2942:             t = getattr(simulation, 'vel_mag_rast_transform', None)
 2943:     except Exception:
 2944:         t = None
 2945:     if t is None:
 2946:         # unit transform
```

## Line 2996
```
 2993:             if out_name in env:
 2994:                 try:
 2995:                     env[out_name][:] = upsampled.astype('f4')
 2996:                 except Exception:
 2997:                     del env[out_name]
 2998:                     env.create_dataset(out_name, data=upsampled.astype('f4'), dtype='f4')
 2999:             else:
```

## Line 3002
```
 2999:             else:
 3000:                 env.create_dataset(out_name, data=upsampled.astype('f4'), dtype='f4')
 3001:             hw.flush()
 3002:         except Exception:
 3003:             # if writing to same file is not desired, just return the upsampled array
 3004:             pass
 3005: 
```

## Line 3011
```
 3008:         with h5py.File(fname, 'r+') as hw2:
 3009:             if tmp_name in hw2:
 3010:                 del hw2[tmp_name]
 3011:     except Exception as e:
 3012:         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=3009)
 3013:         pass
 3014: 
```

## Line 3079
```
 3076:     # export record results to excel via pandas
 3077:     try:
 3078:         logger.info('exporting records to excel...')
 3079:     except Exception as e:
 3080:         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=3076)
 3081:         pass
 3082:     
```

## Line 3094
```
 3091:     
 3092:     try:
 3093:         logger.info('records exported. check output excel file: %s', output_excel)
 3094:     except Exception as e:
 3095:         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=3090)
 3096:         pass
 3097:     
```

## Line 3164
```
 3161:                     
 3162:                 try:
 3163:                     logger.info('Time Step %s complete', i)
 3164:                 except Exception as e:
 3165:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=3159)
 3166:                     pass
 3167: 
```

## Line 3370
```
 3367:         # Attempt to initialize PID plane parameters; fall back to safe defaults
 3368:         try:
 3369:             self.interp_PID()
 3370:         except Exception:
 3371:             # Default: P = 1.0 constant, I = 0, D = 0
 3372:             self.P_params = np.array([0.0, 0.0, 1.0])
 3373:             self.I_params = np.array([0.0, 0.0, 0.0])
```

## Line 3460
```
 3457:             self.P_params, _ = curve_fit(plane_model, (length, velocity), P)
 3458:             self.I_params, _ = curve_fit(plane_model, (length, velocity), I)
 3459:             self.D_params, _ = curve_fit(plane_model, (length, velocity), D)
 3460:         except Exception:
 3461:             # Fallback to constant gains: P=1, I=0, D=0
 3462:             self.P_params = (0.0, 0.0, 1.0)
 3463:             self.I_params = (0.0, 0.0, 0.0)
```

## Line 3879
```
 3876:             
 3877:                 try:
 3878:                     logger.info('running individual %d of generation %d', i+1, generation+1)
 3879:                 except Exception as e:
 3880:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=3879)
 3881:                     pass
 3882:                 
```

## Line 3890
```
 3887:                 
 3888:                 try:
 3889:                     logger.info('P: %0.3f, I: %0.3f, D: %0.3f', self.p[i], self.i[i], self.d[i])
 3890:                 except Exception as e:
 3891:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=3889)
 3892:                     pass
 3893:                 
```

## Line 3918
```
 3915:                             k_d = self.d[i], # k_d
 3916:                             )
 3917:                     
 3918:                 except Exception as e:
 3919:                     try:
 3920:                         logger.exception('PID run failed for P=%0.3f I=%0.3f D=%0.3f: %s', self.p[i], self.i[i], self.d[i], e)
 3921:                     except Exception as e:
```

## Line 3921
```
 3918:                 except Exception as e:
 3919:                     try:
 3920:                         logger.exception('PID run failed for P=%0.3f I=%0.3f D=%0.3f: %s', self.p[i], self.i[i], self.d[i], e)
 3921:                     except Exception as e:
 3922:                         try:
 3923:                             logger.exception('Error while logging in collision metrics catch: %s', e)
 3924:                         except Exception as e:
```

## Line 3924
```
 3921:                     except Exception as e:
 3922:                         try:
 3923:                             logger.exception('Error while logging in collision metrics catch: %s', e)
 3924:                         except Exception as e:
 3925:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=3922)
 3926:                             pass
 3927:                     pop_error_array.append(sim.error_array)
```

## Line 3955
```
 3952:             
 3953:             try:
 3954:                 logger.info('completed generation %d', generation+1)
 3955:             except Exception as e:
 3956:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=3952)
 3957:                 pass
 3958:             
```

## Line 4049
```
 4046:                     from emergent.io.log_writer import LogWriter
 4047:                     self._log_writer = LogWriter(log_dir)
 4048:                     self._memmap_writer = None
 4049:             except Exception:
 4050:                 self._log_writer = None
 4051:                 self._memmap_writer = None
 4052: 
```

## Line 4092
```
 4089:         # (no prints) to reduce console noise.
 4090:         try:
 4091:             from .sockeye import PID_controller
 4092:         except Exception:
 4093:             # fallback: try absolute import path
 4094:             try:
 4095:                 from emergent.salmon_abm.sockeye import PID_controller
```

## Line 4096
```
 4093:             # fallback: try absolute import path
 4094:             try:
 4095:                 from emergent.salmon_abm.sockeye import PID_controller
 4096:             except Exception:
 4097:                 PID_controller = None
 4098: 
 4099:         if PID_controller is not None:
```

## Line 4106
```
 4103:                 if hasattr(self.pid_controller, 'interp_PID'):
 4104:                     try:
 4105:                         self.pid_controller.interp_PID()
 4106:                     except Exception as e:
 4107:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4102)
 4108:                         pass
 4109:             except Exception:
```

## Line 4109
```
 4106:                     except Exception as e:
 4107:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4102)
 4108:                         pass
 4109:             except Exception:
 4110:                 # silently ignore PID attach failures to avoid noisy logs
 4111:                 self.pid_controller = None
 4112:         
```

## Line 4225
```
 4222:             except (OSError, IOError, ValueError, TypeError) as e:
 4223:                 try:
 4224:                     logger.exception('Failed to create MemmapLogWriter (runtime); disabling memmap logging: %s', e)
 4225:                 except Exception as e:
 4226:                     try:
 4227:                         logger.exception('Failed while logging dry/shallow counts block: %s', e)
 4228:                     except Exception as e:
```

## Line 4228
```
 4225:                 except Exception as e:
 4226:                     try:
 4227:                         logger.exception('Failed while logging dry/shallow counts block: %s', e)
 4228:                     except Exception as e:
 4229:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4223)
 4230:                         pass
 4231:                 self._memmap_writer = None
```

## Line 4233
```
 4230:                         pass
 4231:                 self._memmap_writer = None
 4232:                 self._memmap_vars = []
 4233:             except Exception:
 4234:                 logger.exception('Unexpected error while creating MemmapLogWriter; re-raising')
 4235:                 raise
 4236:         self.thrust = self.arr.zeros(num_agents)         # computed theoretical thrust Lighthill 
```

## Line 4263
```
 4260:                     except (OSError, IOError, KeyError, ValueError, TypeError) as e:
 4261:                         try:
 4262:                             logger.exception("Failed to load environment raster '%s' into _env_cache' (runtime): %s", name, e)
 4263:                         except Exception as e:
 4264:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4263)
 4265:                             pass
 4266:                         self._env_cache[name] = None
```

## Line 4267
```
 4264:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4263)
 4265:                             pass
 4266:                         self._env_cache[name] = None
 4267:                     except Exception:
 4268:                         logger.exception("Unexpected error while loading environment raster '%s'; re-raising", name)
 4269:                         raise
 4270:         except (OSError, IOError, AttributeError) as e:
```

## Line 4273
```
 4270:         except (OSError, IOError, AttributeError) as e:
 4271:             try:
 4272:                 logger.exception('Failed to access HDF environment group during init (runtime): %s', e)
 4273:             except Exception as e:
 4274:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4272)
 4275:                 pass
 4276:             self._env_cache = {}
```

## Line 4277
```
 4274:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4272)
 4275:                 pass
 4276:             self._env_cache = {}
 4277:         except Exception:
 4278:             logger.exception('Unexpected error while initializing _env_cache; re-raising')
 4279:             raise
 4280: 
```

## Line 4290
```
 4287:         except (ValueError, TypeError, OSError, RuntimeError) as e:
 4288:             try:
 4289:                 logger.exception("_numba_warmup failed during init warmup (runtime): %s", e)
 4290:             except Exception as e:
 4291:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4288)
 4292:                 pass
 4293:         except Exception:
```

## Line 4293
```
 4290:             except Exception as e:
 4291:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4288)
 4292:                 pass
 4293:         except Exception:
 4294:             logger.exception('_numba_warmup failed during init warmup with unexpected error; re-raising')
 4295:             raise
 4296: 
```

## Line 4304
```
 4301:         except (ValueError, TypeError, OSError, RuntimeError) as e:
 4302:             try:
 4303:                 logger.exception("_numba_warmup_for_sim failed during sim warmup (runtime): %s", e)
 4304:             except Exception as e:
 4305:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4301)
 4306:                 pass
 4307:         except Exception:
```

## Line 4307
```
 4304:             except Exception as e:
 4305:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4301)
 4306:                 pass
 4307:         except Exception:
 4308:             logger.exception('_numba_warmup_for_sim failed during sim warmup with unexpected error; re-raising')
 4309:             raise
 4310: 
```

## Line 4319
```
 4316:             # default config (can be overridden by setting these attrs before init)
 4317:             factor = getattr(self, 'alongstream_factor', 4)
 4318:             create_on_init = getattr(self, 'create_alongstream_on_init', True)
 4319:         except Exception:
 4320:             factor = 4
 4321:             create_on_init = True
 4322: 
```

## Line 4345
```
 4342:                         try:
 4343:                             # attempt to map commonly used rasters from HECRAS onto our HDF grid
 4344:                             map_hecras_to_env_rasters(self, hecras_hdf, raster_names=['depth','wetted','vel_x','vel_y'])
 4345:                         except Exception:
 4346:                             try:
 4347:                                 logger.exception("map_hecras_to_env_rasters failed for %s, will try cached load", hecras_hdf)
 4348:                             except Exception as e:
```

## Line 4348
```
 4345:                         except Exception:
 4346:                             try:
 4347:                                 logger.exception("map_hecras_to_env_rasters failed for %s, will try cached load", hecras_hdf)
 4348:                             except Exception as e:
 4349:                                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4344)
 4350:                                 pass
 4351:                             # fallback: try loading HECRAS plan into cache (KDTree) and then map
```

## Line 4355
```
 4352:                             try:
 4353:                                 load_hecras_plan_cached(self, hecras_hdf)
 4354:                                 map_hecras_to_env_rasters(self, hecras_hdf, raster_names=['depth','wetted','vel_x','vel_y'])
 4355:                             except Exception:
 4356:                                 try:
 4357:                                     logger.exception("load_hecras_plan_cached or second map_hecras_to_env_rasters attempt failed for %s", hecras_hdf)
 4358:                                 except Exception as e:
```

## Line 4358
```
 4355:                             except Exception:
 4356:                                 try:
 4357:                                     logger.exception("load_hecras_plan_cached or second map_hecras_to_env_rasters attempt failed for %s", hecras_hdf)
 4358:                                 except Exception as e:
 4359:                                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4353)
 4360:                                     pass
 4361: 
```

## Line 4365
```
 4362:                         # compute coarsened alongstream raster using created env rasters
 4363:                         try:
 4364:                             compute_coarsened_alongstream_raster(self, factor=factor, outlet_xy=None, depth_name='depth', wetted_name='wetted', out_name='along_stream_dist')
 4365:                         except Exception:
 4366:                             try:
 4367:                                 logger.exception("compute_coarsened_alongstream_raster failed, will try compute_alongstream_raster")
 4368:                             except Exception as e:
```

## Line 4368
```
 4365:                         except Exception:
 4366:                             try:
 4367:                                 logger.exception("compute_coarsened_alongstream_raster failed, will try compute_alongstream_raster")
 4368:                             except Exception as e:
 4369:                                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4368)
 4370:                                 pass
 4371:                             try:
```

## Line 4373
```
 4370:                                 pass
 4371:                             try:
 4372:                                 compute_alongstream_raster(self, outlet_xy=None, depth_name='depth', wetted_name='wetted', out_name='along_stream_dist')
 4373:                             except Exception:
 4374:                                 try:
 4375:                                     logger.exception("compute_alongstream_raster also failed")
 4376:                                 except Exception as e:
```

## Line 4376
```
 4373:                             except Exception:
 4374:                                 try:
 4375:                                     logger.exception("compute_alongstream_raster also failed")
 4376:                                 except Exception as e:
 4377:                                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4375)
 4378:                                     pass
 4379:                     except Exception:
```

## Line 4379
```
 4376:                                 except Exception as e:
 4377:                                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4375)
 4378:                                     pass
 4379:                     except Exception:
 4380:                         try:
 4381:                             logger.exception("mapping external HDF failed, will try computing alongstream on current hdf5")
 4382:                         except Exception as e:
```

## Line 4382
```
 4379:                     except Exception:
 4380:                         try:
 4381:                             logger.exception("mapping external HDF failed, will try computing alongstream on current hdf5")
 4382:                         except Exception as e:
 4383:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4380)
 4384:                             pass
 4385:                         # if reading external HDF fails, try computing on current hdf5
```

## Line 4388
```
 4385:                         # if reading external HDF fails, try computing on current hdf5
 4386:                         try:
 4387:                             compute_coarsened_alongstream_raster(self, factor=factor)
 4388:                         except Exception:
 4389:                             try:
 4390:                                 logger.exception("compute_coarsened_alongstream_raster fallback failed")
 4391:                             except Exception as e:
```

## Line 4391
```
 4388:                         except Exception:
 4389:                             try:
 4390:                                 logger.exception("compute_coarsened_alongstream_raster fallback failed")
 4391:                             except Exception as e:
 4392:                                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4388)
 4393:                                 pass
 4394:                 else:
```

## Line 4398
```
 4395:                     # compute on in-project HDF if environment rasters were loaded
 4396:                     try:
 4397:                         compute_coarsened_alongstream_raster(self, factor=factor)
 4398:                     except Exception:
 4399:                             try:
 4400:                                 logger.exception("compute_coarsened_alongstream_raster failed when computing on in-project HDF")
 4401:                             except Exception as e:
```

## Line 4401
```
 4398:                     except Exception:
 4399:                             try:
 4400:                                 logger.exception("compute_coarsened_alongstream_raster failed when computing on in-project HDF")
 4401:                             except Exception as e:
 4402:                                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4397)
 4403:                                 pass
 4404:             except Exception as e:
```

## Line 4404
```
 4401:                             except Exception as e:
 4402:                                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4397)
 4403:                                 pass
 4404:             except Exception as e:
 4405:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4399)
 4406:                 pass
 4407:         
```

## Line 4435
```
 4432:             if os.path.exists(path):
 4433:                 try:
 4434:                     logger.info("Importing %s from %s as %s", key, path, surface_type)
 4435:                 except Exception as e:
 4436:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4435)
 4437:                     pass
 4438:                 self.enviro_import(path, surface_type)
```

## Line 4441
```
 4438:                 self.enviro_import(path, surface_type)
 4439:                 try:
 4440:                     logger.info("Successfully imported %s", key)
 4441:                 except Exception as e:
 4442:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4440)
 4443:                     pass
 4444:             else:
```

## Line 4447
```
 4444:             else:
 4445:                 try:
 4446:                     logger.warning("Raster file not found: %s", path)
 4447:                 except Exception as e:
 4448:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4445)
 4449:                     pass
 4450: 
```

## Line 4472
```
 4469:                 logger.info("%s", "\n" + "="*80)
 4470:                 logger.info("INITIALIZING HECRAS MODE")
 4471:                 logger.info("%s", "="*80)
 4472:             except Exception as e:
 4473:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4469)
 4474:                 pass
 4475:             
```

## Line 4496
```
 4493:                     centerline_derived = True
 4494:                     try:
 4495:                         logger.info("Using HECRAS-derived centerline (%0.2fm)", self.centerline.length)
 4496:                     except Exception as e:
 4497:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4492)
 4498:                         pass
 4499:                 
```

## Line 4508
```
 4505:                     y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
 4506:                     try:
 4507:                         logger.info("HECRAS extent: X=[%0.2f, %0.2f], Y=[%0.2f, %0.2f]", x_min, x_max, y_min, y_max)
 4508:                     except Exception as e:
 4509:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4503)
 4510:                         pass
 4511: 
```

## Line 4528
```
 4525:                                 perim_timestep = max(0, num_ts // 2)
 4526:                             else:
 4527:                                 perim_timestep = 0
 4528:                         except Exception:
 4529:                             perim_timestep = 0
 4530: 
 4531:                     # Use helper to infer wetted perimeter from the HECRAS plan file
```

## Line 4534
```
 4531:                     # Use helper to infer wetted perimeter from the HECRAS plan file
 4532:                     try:
 4533:                         perim_info = infer_wetted_perimeter_from_hecras(self.hecras_plan_path, depth_threshold=perim_depth, timestep=perim_timestep, verbose=False)
 4534:                     except Exception:
 4535:                         perim_info = None
 4536: 
 4537:                     if perim_info is not None:
```

## Line 4558
```
 4555:                                 else:
 4556:                                     # Unknown shape -> treat as no perimeter
 4557:                                     perim_info = {'perimeter_points': np.zeros((0,2)), 'perimeter_cells': np.zeros((0,), dtype=int), 'wetted_mask': None}
 4558:                             except Exception:
 4559:                                 perim_info = {'perimeter_points': np.zeros((0,2)), 'perimeter_cells': np.zeros((0,), dtype=int), 'wetted_mask': None}
 4560: 
 4561:                         # attach raw perimeter points and masks
```

## Line 4570
```
 4567:                             # prefer alpha-shape (concave hull) produced by tin_helpers
 4568:                             try:
 4569:                                 from emergent.salmon_abm.tin_helpers import alpha_shape
 4570:                             except Exception:
 4571:                                 try:
 4572:                                     from .tin_helpers import alpha_shape
 4573:                                 except Exception:
```

## Line 4573
```
 4570:                             except Exception:
 4571:                                 try:
 4572:                                     from .tin_helpers import alpha_shape
 4573:                                 except Exception:
 4574:                                     alpha_shape = None
 4575: 
 4576:                             if alpha_shape is not None and self.perimeter_points.shape[0] > 3:
```

## Line 4584
```
 4581:                                 from shapely.geometry import MultiPoint
 4582:                                 mp = MultiPoint([tuple(p) for p in self.perimeter_points])
 4583:                                 self.perimeter_polygon = mp.convex_hull
 4584:                         except Exception:
 4585:                             self.perimeter_polygon = None
 4586: 
 4587:                         try:
```

## Line 4589
```
 4586: 
 4587:                         try:
 4588:                             logger.info("HECRAS perimeter: %d points; polygon=%s", len(self.perimeter_points), 'yes' if self.perimeter_polygon is not None else 'no')
 4589:                         except Exception as e:
 4590:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4589)
 4591:                             pass
 4592:                     else:
```

## Line 4597
```
 4594:                         self.perimeter_cells = np.zeros((0,), dtype=int)
 4595:                         self.wetted_mask = None
 4596:                         self.perimeter_polygon = None
 4597:                 except Exception as e:
 4598:                     try:
 4599:                         logger.warning("Failed to compute HECRAS perimeter: %s", e)
 4600:                     except Exception as e:
```

## Line 4600
```
 4597:                 except Exception as e:
 4598:                     try:
 4599:                         logger.warning("Failed to compute HECRAS perimeter: %s", e)
 4600:                     except Exception as e:
 4601:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4599)
 4602:                         pass
 4603:                     self.perimeter_points = np.zeros((0,2))
```

## Line 4611
```
 4608:             except (OSError, IOError, KeyError, ValueError) as e:
 4609:                 try:
 4610:                     logger.exception("HECRAS initialization failed due to IO/Key/Value error: %s", e)
 4611:                 except Exception as e:
 4612:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4609)
 4613:                     pass
 4614:                 # Fall back to loading KDTree only
```

## Line 4617
```
 4614:                 # Fall back to loading KDTree only
 4615:                 try:
 4616:                     load_hecras_plan_cached(self, self.hecras_plan_path, field_names=self.hecras_fields)
 4617:                 except Exception as e2:
 4618:                     # Disable HECRAS mode to avoid inconsistent later assumptions
 4619:                     try:
 4620:                         logger.warning("load_hecras_plan_cached failed; disabling HECRAS mode: %s", e2)
```

## Line 4621
```
 4618:                     # Disable HECRAS mode to avoid inconsistent later assumptions
 4619:                     try:
 4620:                         logger.warning("load_hecras_plan_cached failed; disabling HECRAS mode: %s", e2)
 4621:                     except Exception as e:
 4622:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4618)
 4623:                         pass
 4624:                     self.hecras_plan_path = None
```

## Line 4627
```
 4624:                     self.hecras_plan_path = None
 4625:                     self.use_hecras = False
 4626:                     self.hecras_mapping_enabled = False
 4627:             except Exception:
 4628:                 # Re-raise unexpected exceptions so we don't mask programming errors
 4629:                 raise
 4630:                     
```

## Line 4656
```
 4653:                     env = None
 4654:                     try:
 4655:                         env = self.hdf5.get('environment') if self.hdf5 is not None else None
 4656:                     except Exception:
 4657:                         env = None
 4658: 
 4659:                     distance_to = None
```

## Line 4673
```
 4670:                                 x_coords = np.array(env['x_coords'])
 4671:                             if 'y_coords' in env:
 4672:                                 y_coords = np.array(env['y_coords'])
 4673:                         except Exception:
 4674:                             distance_to = None
 4675:                             wetted = None
 4676: 
```

## Line 4690
```
 4687:                                     x_coords = x_coords if x_coords is not None else np.array(env2['x_coords'])
 4688:                                 if env2 is not None and 'y_coords' in env2:
 4689:                                     y_coords = y_coords if y_coords is not None else np.array(env2['y_coords'])
 4690:                         except Exception as e:
 4691:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4686)
 4692:                             pass
 4693: 
```

## Line 4704
```
 4701:                             pix = getattr(self, 'depth_rast_transform', None)
 4702:                             pixel_width = pix.a if pix is not None else 1.0
 4703:                             distance_to_rast = distance_transform_edt(mask) * pixel_width
 4704:                         except Exception:
 4705:                             distance_to_rast = None
 4706: 
 4707:                     if distance_to_rast is None or distance_to_rast.size == 0:
```

## Line 4737
```
 4734:                                             with h5py.File(self.hecras_plan_path, 'r') as ph:
 4735:                                                 hecras_coords = ph['/Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:]
 4736:                                             target_affine = compute_affine_from_hecras(hecras_coords)
 4737:                                         except Exception:
 4738:                                             target_affine = None
 4739:                                 except Exception:
 4740:                                     target_affine = None
```

## Line 4739
```
 4736:                                             target_affine = compute_affine_from_hecras(hecras_coords)
 4737:                                         except Exception:
 4738:                                             target_affine = None
 4739:                                 except Exception:
 4740:                                     target_affine = None
 4741: 
 4742:                             main_centerline, all_lines = derive_centerline_from_distance_raster(distance_to_rast, transform=target_affine, footprint_size=5, min_length=50)
```

## Line 4747
```
 4744:                                 self.centerline = self.centerline_import(main_centerline)
 4745:                                 try:
 4746:                                     logger.info('Derived centerline from HECRAS rasters (skeletonized)')
 4747:                                 except Exception as e:
 4748:                                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4742)
 4749:                                     pass
 4750:                                 centerline_derived = True
```

## Line 4758
```
 4755:                                     centerline_derived = True
 4756:                                 else:
 4757:                                     raise RuntimeError('Extracted centerline is invalid or too short')
 4758:                         except Exception as e:
 4759:                             raise
 4760:                 except Exception as e:
 4761:                     try:
```

## Line 4760
```
 4757:                                     raise RuntimeError('Extracted centerline is invalid or too short')
 4758:                         except Exception as e:
 4759:                             raise
 4760:                 except Exception as e:
 4761:                     try:
 4762:                         logger.warning('Could not derive centerline from HECRAS: %s', e)
 4763:                     except Exception as e:
```

## Line 4763
```
 4760:                 except Exception as e:
 4761:                     try:
 4762:                         logger.warning('Could not derive centerline from HECRAS: %s', e)
 4763:                     except Exception as e:
 4764:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4763)
 4765:                         pass
 4766:                     # fall back to file-based import only if a valid file was provided
```

## Line 4818
```
 4815:                     except (KeyError, IndexError, OSError) as e:
 4816:                         try:
 4817:                             logger.warning('HECRAS depth dataset missing or unreadable: %s', e)
 4818:                         except Exception as e:
 4819:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4817)
 4820:                             pass
 4821:                     try:
```

## Line 4827
```
 4824:                     except (KeyError, IndexError, OSError) as e:
 4825:                         try:
 4826:                             logger.warning('HECRAS velocity datasets missing or unreadable: %s', e)
 4827:                         except Exception as e:
 4828:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4825)
 4829:                             pass
 4830:                 
```

## Line 4856
```
 4853:                         self.y_vel = self.apply_hecras_mapping(node_fields['vel_y'])
 4854:                         self.vel_mag = np.sqrt(self.x_vel**2 + self.y_vel**2)
 4855:                     
 4856:             except Exception as e:
 4857:                 try:
 4858:                     logger.warning("Failed to initialize HECRAS environment: %s", e)
 4859:                 except Exception as e:
```

## Line 4859
```
 4856:             except Exception as e:
 4857:                 try:
 4858:                     logger.warning("Failed to initialize HECRAS environment: %s", e)
 4859:                 except Exception as e:
 4860:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4856)
 4861:                     pass
 4862:         
```

## Line 4892
```
 4889:             keys = ', '.join(sorted(wdict.keys()))
 4890:             try:
 4891:                 logger.info("Applied behavioral weights (keys): %s", keys)
 4892:             except Exception as e:
 4893:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4888)
 4894:                 pass
 4895:         except Exception:
```

## Line 4895
```
 4892:             except Exception as e:
 4893:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4888)
 4894:                 pass
 4895:         except Exception:
 4896:             try:
 4897:                 logger.warning('Applied behavioral weights (some fields may be unavailable)')
 4898:             except Exception as e:
```

## Line 4898
```
 4895:         except Exception:
 4896:             try:
 4897:                 logger.warning('Applied behavioral weights (some fields may be unavailable)')
 4898:             except Exception as e:
 4899:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4893)
 4900:                 pass
 4901:         # Numba warm-up: call compiled loops with tiny inputs to trigger JIT at initialization
```

## Line 4916
```
 4913:                 except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
 4914:                     try:
 4915:                         logger.exception("_compute_schooling_loop warmup failed (runtime): %s", e)
 4916:                     except Exception as e:
 4917:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4916)
 4918:                         pass
 4919:                 except Exception:
```

## Line 4919
```
 4916:                     except Exception as e:
 4917:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4916)
 4918:                         pass
 4919:                 except Exception:
 4920:                     logger.exception('_compute_schooling_loop warmup failed with unexpected error; re-raising')
 4921:                     raise
 4922:                 dummy_drag = np.zeros(2, dtype=np.float64)
```

## Line 4928
```
 4925:                 except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
 4926:                     try:
 4927:                         logger.exception("_compute_drafting_loop warmup failed (runtime): %s", e)
 4928:                     except Exception as e:
 4929:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4927)
 4930:                         pass
 4931:                 except Exception:
```

## Line 4931
```
 4928:                     except Exception as e:
 4929:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4927)
 4930:                         pass
 4931:                 except Exception:
 4932:                     logger.exception('_compute_drafting_loop warmup failed with unexpected error; re-raising')
 4933:                     raise
 4934:         except Exception:
```

## Line 4934
```
 4931:                 except Exception:
 4932:                     logger.exception('_compute_drafting_loop warmup failed with unexpected error; re-raising')
 4933:                     raise
 4934:         except Exception:
 4935:             try:
 4936:                 logger.exception("apply_behavioral_weights: unexpected failure during numba warmup block")
 4937:             except Exception as e:
```

## Line 4937
```
 4934:         except Exception:
 4935:             try:
 4936:                 logger.exception("apply_behavioral_weights: unexpected failure during numba warmup block")
 4937:             except Exception as e:
 4938:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4935)
 4939:                 pass
 4940:     
```

## Line 4997
```
 4994:         try:
 4995:             # Add small random heading perturbation uniform [-pi, pi]
 4996:             self.heading = np.random.uniform(-np.pi, np.pi, size=self.num_agents)
 4997:         except Exception as e:
 4998:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=4994)
 4999:             pass
 5000:         try:
```

## Line 5012
```
 5009:                 high = 1.0 + float(frac)
 5010:                 self.sog = base * np.random.uniform(low, high, size=self.num_agents)
 5011:                 self.ideal_sog = self.sog.copy()
 5012:         except Exception as e:
 5013:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5008)
 5014:             pass
 5015: 
```

## Line 5026
```
 5023:                 self.y_vel = np.zeros(self.num_agents)
 5024:             self.x_vel = self.x_vel + vel_jitter[:, 0]
 5025:             self.y_vel = self.y_vel + vel_jitter[:, 1]
 5026:         except Exception as e:
 5027:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5021)
 5028:             pass
 5029: 
```

## Line 5042
```
 5039:                         try:
 5040:                             newv = float(v) + np.random.normal(0, std)
 5041:                             setattr(bw, k, newv)
 5042:                         except Exception as e:
 5043:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5042)
 5044:                             pass
 5045:                 # apply mutated weights
```

## Line 5048
```
 5045:                 # apply mutated weights
 5046:                 try:
 5047:                     self.apply_behavioral_weights(bw)
 5048:                 except Exception as e:
 5049:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5047)
 5050:                     pass
 5051:         except Exception as e:
```

## Line 5051
```
 5048:                 except Exception as e:
 5049:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5047)
 5050:                     pass
 5051:         except Exception as e:
 5052:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5049)
 5053:             pass
 5054:             self.Y = self._initial_positions['Y'].copy()
```

## Line 5108
```
 5105:             if hasattr(self, '_kdtree_cache'):
 5106:                 try:
 5107:                     del self._kdtree_cache
 5108:                 except Exception:
 5109:                     self._kdtree_cache = {}
 5110:         except Exception as e:
 5111:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5107)
```

## Line 5110
```
 5107:                     del self._kdtree_cache
 5108:                 except Exception:
 5109:                     self._kdtree_cache = {}
 5110:         except Exception as e:
 5111:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5107)
 5112:             pass
 5113:         try:
```

## Line 5117
```
 5114:             if hasattr(self, '_hecras_map'):
 5115:                 try:
 5116:                     del self._hecras_map
 5117:                 except Exception:
 5118:                     self._hecras_map = None
 5119:         except Exception as e:
 5120:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5115)
```

## Line 5119
```
 5116:                     del self._hecras_map
 5117:                 except Exception:
 5118:                     self._hecras_map = None
 5119:         except Exception as e:
 5120:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5115)
 5121:             pass
 5122:         # Drop large historical buffers if present to reclaim memory
```

## Line 5128
```
 5125:                 self.swim_speeds = np.full((self.num_agents, 1), np.nan)
 5126:             if hasattr(self, 'past_centerline_meas'):
 5127:                 self.past_centerline_meas = np.full((self.num_agents, 1), np.nan)
 5128:         except Exception as e:
 5129:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5123)
 5130:             pass
 5131:         # Force garbage collection to free memory immediately
```

## Line 5135
```
 5132:         try:
 5133:             import gc
 5134:             gc.collect()
 5135:         except Exception as e:
 5136:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5135)
 5137:             pass
 5138:         
```

## Line 5155
```
 5152:         
 5153:         try:
 5154:             logger.info("Spatial state reset complete. Behavioral weights preserved.")
 5155:         except Exception as e:
 5156:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5154)
 5157:             pass
 5158:     
```

## Line 5296
```
 5293:             agent_data.create_dataset("dist_per_bout", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
 5294:             agent_data.create_dataset("bout_dur", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
 5295:             agent_data.create_dataset("time_of_jump", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
 5296:         except Exception:
 5297:             try:
 5298:                 logger.exception("initialize_hdf5: failed while creating HDF5 datasets")
 5299:             except Exception as e:
```

## Line 5299
```
 5296:         except Exception:
 5297:             try:
 5298:                 logger.exception("initialize_hdf5: failed while creating HDF5 datasets")
 5299:             except Exception as e:
 5300:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5297)
 5301:                 pass
 5302:             raise
```

## Line 5362
```
 5359:                             from emergent.io.log_writer_memmap import MemmapLogWriter
 5360:                             out_dir = self._memmap_config.get('out_dir', os.path.join(self.model_dir, 'logs', 'deferred'))
 5361:                             self._memmap_writer = MemmapLogWriter(out_dir, var_shapes, dtype=np.float32)
 5362:                         except Exception:
 5363:                             self._memmap_writer = None
 5364: 
 5365:                     # write buffered timesteps using memmap writer if available
```

## Line 5372
```
 5369:                         for k in self._hdf5_buffers.keys():
 5370:                             try:
 5371:                                 arrays_2d[k] = self._hdf5_buffers[k][:, :write_len].astype('f4')
 5372:                             except Exception:
 5373:                                 arrays_2d[k] = np.zeros((self.num_agents, write_len), dtype='f4')
 5374:                         try:
 5375:                             self._memmap_writer.append_block(t_start, arrays_2d)
```

## Line 5376
```
 5373:                                 arrays_2d[k] = np.zeros((self.num_agents, write_len), dtype='f4')
 5374:                         try:
 5375:                             self._memmap_writer.append_block(t_start, arrays_2d)
 5376:                         except Exception:
 5377:                             # fallback to per-step append on failure
 5378:                             for offset in range(write_len):
 5379:                                 t_idx = t_start + offset
```

## Line 5383
```
 5380:                                 arrays = {k: self._hdf5_buffers[k][:, offset].astype('f4') for k in self._hdf5_buffers.keys()}
 5381:                                 try:
 5382:                                     self._memmap_writer.append(t_idx, arrays)
 5383:                                 except Exception as e:
 5384:                                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5380)
 5385:                                     pass
 5386:                     elif getattr(self, '_log_writer', None) is not None:
```

## Line 5393
```
 5390:                             arrays = {k: self._hdf5_buffers[k][:, offset].astype('f4') for k in self._hdf5_buffers.keys()}
 5391:                             try:
 5392:                                 self._log_writer.append(t_idx, arrays)
 5393:                             except Exception as e:
 5394:                                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5389)
 5395:                                 pass
 5396:                 else:
```

## Line 5402
```
 5399:                         if ds_name in self.hdf5:
 5400:                             try:
 5401:                                 self.hdf5[ds_name][:, t_start:t_end+1] = buf[:, :write_len]
 5402:                             except Exception:
 5403:                                 for offset in range(write_len):
 5404:                                     self.hdf5[ds_name][:, t_start + offset] = buf[:, offset]
 5405:                 # reset buffer
```

## Line 5444
```
 5441:                             existing_r[mask] = acc_arr[mask]
 5442:                             dsr[:, :] = existing_r
 5443:                             acc_arr.fill(0)
 5444:                     except Exception as e:
 5445:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5439)
 5446:                         pass
 5447:                 except Exception:
```

## Line 5447
```
 5444:                     except Exception as e:
 5445:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5439)
 5446:                         pass
 5447:                 except Exception:
 5448:                     # If memory group doesn't exist or write fails, skip silently
 5449:                     pass
 5450:                 self.hdf5.flush()
```

## Line 5495
```
 5492:         except (OSError, IOError) as e:
 5493:             try:
 5494:                 logger.exception("Failed to open raster %s: %s", data_dir, e)
 5495:             except Exception as e:
 5496:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5495)
 5497:                 pass
 5498:             return
```

## Line 5520
```
 5517:         if 'x_coords' not in self.hdf5:
 5518:             try:
 5519:                 logger.info("Creating x_coords and y_coords with dimensions: height=%s, width=%s, rows=%s, cols=%s", height, width, src.shape[0], src.shape[1])
 5520:             except Exception as e:
 5521:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5519)
 5522:                 pass
 5523: 
```

## Line 5537
```
 5534:             except (OSError, IOError, ValueError) as e:
 5535:                 try:
 5536:                     logger.exception('Failed to create x_coords/y_coords datasets: %s', e)
 5537:                 except Exception as e:
 5538:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5535)
 5539:                     pass
 5540:                 raise
```

## Line 5544
```
 5541: 
 5542:                 try:
 5543:                     logger.info("Created datasets with shape: %s", dset_x.shape)
 5544:                 except Exception as e:
 5545:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5541)
 5546:                     pass
 5547: 
```

## Line 5562
```
 5559:                     except (OSError, IOError, ValueError, IndexError) as e:
 5560:                         try:
 5561:                             logger.exception('Failed writing x/y coords chunk starting at row %s: %s', i, e)
 5562:                         except Exception as e:
 5563:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5558)
 5564:                             pass
 5565:                         raise
```

## Line 5573
```
 5570:                 except (OSError, IOError) as e:
 5571:                     try:
 5572:                         logger.exception('Failed to flush HDF5 after writing x/y coords: %s', e)
 5573:                     except Exception as e:
 5574:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5568)
 5575:                         pass
 5576:                     raise
```

## Line 5580
```
 5577: 
 5578:                 try:
 5579:                     logger.info("Successfully created x_coords and y_coords; flushed to disk. Shape: %s, first value: %s", dset_x.shape, dset_x[0,0])
 5580:                 except Exception as e:
 5581:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5580)
 5582:                     pass
 5583:         else:
```

## Line 5586
```
 5583:         else:
 5584:             try:
 5585:                 logger.info("x_coords already exists with shape: %s", self.hdf5['x_coords'].shape)
 5586:             except Exception as e:
 5587:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5585)
 5588:                 pass
 5589: 
```

## Line 5800
```
 5797:                     dists, inds = tree.query(pts, k=1)
 5798:                     rows = (inds // xs.shape[1]).astype(int)
 5799:                     cols = (inds % xs.shape[1]).astype(int)
 5800:                 except Exception:
 5801:                     logger.exception('Unexpected error during geo->pixel conversion; re-raising')
 5802:                     raise
 5803:                 # clamp
```

## Line 5809
```
 5806:                 vals = rast[rows, cols]
 5807:                 # reshape to agent grid
 5808:                 return vals.reshape(self.X.shape)
 5809:         except Exception as e:
 5810:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5807)
 5811:             pass
 5812: 
```

## Line 5820
```
 5817:             except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
 5818:                 try:
 5819:                     logger.exception('Numba project-points helper failed (runtime); falling back to numpy: %s', e)
 5820:                 except Exception as e:
 5821:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5817)
 5822:                     pass
 5823:                 dists = _project_points_onto_line_numpy(xs_line, ys_line, px, py)
```

## Line 5824
```
 5821:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5817)
 5822:                     pass
 5823:                 dists = _project_points_onto_line_numpy(xs_line, ys_line, px, py)
 5824:             except Exception:
 5825:                 logger.exception('Unexpected error in project-points numba helper; re-raising')
 5826:                 raise
 5827:         else:
```

## Line 5841
```
 5838:         if self.use_hecras and hasattr(self, '_hecras_geometry_info'):
 5839:             try:
 5840:                 logger.info("Using HECRAS distance-to-bank (already computed)")
 5841:             except Exception as e:
 5842:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5837)
 5843:                 pass
 5844:             # distance_to_bank was already written to HDF5 by initialize_hecras_geometry
```

## Line 5854
```
 5851:             if env is None or 'wetted' not in env:
 5852:                 try:
 5853:                     logger.warning("No wetted raster found, skipping boundary_surface")
 5854:                 except Exception as e:
 5855:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=5849)
 5856:                     pass
 5857:                 return
```

## Line 5862
```
 5859:             # Additional guard: ensure raster has valid shape
 5860:             if raster.size == 0 or raster.shape[0] == 0 or raster.shape[1] == 0:
 5861:                 return
 5862:         except Exception:
 5863:             return
 5864: 
 5865:         # pixel width fallback
```

## Line 5868
```
 5865:         # pixel width fallback
 5866:         try:
 5867:             pixel_width = self.depth_rast_transform.a
 5868:         except Exception:
 5869:             pixel_width = 1.0
 5870: 
 5871:         # Compute the Euclidean distance transform. This computes the distance to the nearest zero (background) for all non-zero (foreground) pixels.
```

## Line 5895
```
 5892:                 h, w = xs.shape
 5893:                 self.height = h
 5894:                 self.width = w
 5895:             except Exception:
 5896:                 # cannot determine shape; skip writing
 5897:                 return
 5898: 
```

## Line 5957
```
 5954:                         yrange = float(coords[:, 1].max() - coords[:, 1].min())
 5955:                         self.width = int(np.ceil(xrange / self.avoid_cell_size)) + 1
 5956:                         self.height = int(np.ceil(yrange / self.avoid_cell_size)) + 1
 5957:             except Exception:
 5958:                 base_transform = compute_affine_from_hecras(np.array([[0.0, 0.0]]), target_cell_size=self.avoid_cell_size)
 5959: 
 5960:         self.mental_map_transform = compute_affine_from_hecras(base_transform * 0.0 if False else np.array([[base_transform.c, base_transform.f]]), target_cell_size=self.avoid_cell_size) if False else base_transform
```

## Line 6012
```
 6009:                     base_t = compute_affine_from_hecras(coords, target_cell_size=self.refugia_cell_size)
 6010:                 else:
 6011:                     base_t = compute_affine_from_hecras(np.array([[0.0, 0.0]]), target_cell_size=self.refugia_cell_size)
 6012:             except Exception:
 6013:                 base_t = compute_affine_from_hecras(np.array([[0.0, 0.0]]), target_cell_size=self.refugia_cell_size)
 6014: 
 6015:         self.refugia_map_transform = base_t
```

## Line 6092
```
 6089:                 if hasattr(self, 'height') and hasattr(self, 'width'):
 6090:                     target_shape = (self.height, self.width)
 6091:                 ensure_hdf_coords_from_hecras(self, self.hecras_plan_path, target_shape=target_shape, target_transform=target_transform)
 6092:             except Exception as e:
 6093:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=6092)
 6094:                 pass
 6095: 
```

## Line 6101
```
 6098:             if getattr(self, 'hecras_write_rasters', False):
 6099:                 try:
 6100:                     map_hecras_to_env_rasters(self, self.hecras_plan_path, field_names=getattr(self, 'hecras_fields', None), k=getattr(self, 'hecras_k', 8))
 6101:                 except Exception as e:
 6102:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=6100)
 6103:                     pass
 6104: 
```

## Line 6129
```
 6126:                             if vals is not None:
 6127:                                 mapped = np.asarray(vals).flatten()
 6128:                                 break
 6129:                         except Exception:
 6130:                             mapped = None
 6131:                             continue
 6132:                 # fallback: fill with NaN
```

## Line 6149
```
 6146:             # prefer using a cached inverse affine to avoid repeated inversion
 6147:             inv = get_inv_transform(self, transforms[0])
 6148:             rows, cols = geo_to_pixel_from_inv(inv, self.X, self.Y)
 6149:         except Exception:
 6150:             rows, cols = geo_to_pixel(self.X, self.Y, transforms[0])
 6151:         rows = np.clip(rows, 0, self.height - 1).astype(int)
 6152:         cols = np.clip(cols, 0, self.width - 1).astype(int)
```

## Line 6339
```
 6336:             try:
 6337:                 from emergent.salmon_abm.sockeye import _safe_build_kdtree
 6338:                 tree = _safe_build_kdtree(hecras_nodes, name='hecras_nodes_tree')
 6339:             except Exception:
 6340:                 tree = None
 6341:         else:
 6342:             tree = self.hecras_map['tree']
```

## Line 6359
```
 6356:                 # gather distances
 6357:                 row_idxs = np.arange(dist_all.shape[0])[:, None]
 6358:                 dists = dist_all[row_idxs, inds]
 6359:             except Exception:
 6360:                 raise RuntimeError('Failed to compute nearest HECRAS nodes (no KDTree and brute-force failed)')
 6361:         # ensure shapes (M,k)
 6362:         if k == 1:
```

## Line 6429
```
 6426:                     try:
 6427:                         from shapely.prepared import prep
 6428:                         prep_poly = prep(poly)
 6429:                     except Exception:
 6430:                         prep_poly = None
 6431: 
 6432:                     wet_arr = np.zeros(len(xs), dtype=float)
```

## Line 6439
```
 6436:                             for i in idxs:
 6437:                                 try:
 6438:                                     wet_arr[i] = 1.0 if prep_poly.contains(Point(float(xs[i]), float(ys[i]))) else 0.0
 6439:                                 except Exception:
 6440:                                     wet_arr[i] = 0.0
 6441:                         else:
 6442:                             # fallback to vectorized check if available
```

## Line 6447
```
 6444:                                 from shapely import vectorized
 6445:                                 mask = vectorized.contains(poly, xs[idxs], ys[idxs])
 6446:                                 wet_arr[idxs] = mask.astype(float)
 6447:                             except Exception:
 6448:                                 # last-resort: per-point contains without prep
 6449:                                 from shapely.geometry import Point
 6450:                                 for i in idxs:
```

## Line 6453
```
 6450:                                 for i in idxs:
 6451:                                     try:
 6452:                                         wet_arr[i] = 1.0 if poly.contains(Point(float(xs[i]), float(ys[i]))) else 0.0
 6453:                                     except Exception:
 6454:                                         wet_arr[i] = 0.0
 6455: 
 6456:                     # reshape into agent grid
```

## Line 6458
```
 6455: 
 6456:                     # reshape into agent grid
 6457:                     self.wet = wet_arr.reshape(self.X.shape)
 6458:                 except Exception:
 6459:                     # fallback: do not change self.wet here
 6460:                     pass
 6461:         except Exception as e:
```

## Line 6461
```
 6458:                 except Exception:
 6459:                     # fallback: do not change self.wet here
 6460:                     pass
 6461:         except Exception as e:
 6462:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=6459)
 6463:             pass
 6464: 
```

## Line 6512
```
 6509:             vals = self.batch_sample_environment([self.vel_x_rast_transform, self.vel_y_rast_transform], ['vel_x', 'vel_y'])
 6510:             self.x_vel = vals.get('vel_x', np.zeros(self.num_agents))
 6511:             self.y_vel = vals.get('vel_y', np.zeros(self.num_agents))
 6512:         except Exception:
 6513:             self.x_vel = np.zeros(self.num_agents)
 6514:             self.y_vel = np.zeros(self.num_agents)
 6515:         
```

## Line 6557
```
 6554:             self.heading = self.arr.where(values < 0,
 6555:                                            (self.arr.radians(360) + values) - self.arr.radians(180),
 6556:                                            values - self.arr.radians(180))
 6557:         except Exception:
 6558:             # Fallback: compute heading from vector velocity components (x_vel, y_vel)
 6559:             try:
 6560:                 vals_deg = np.degrees(np.arctan2(self.y_vel, self.x_vel))
```

## Line 6561
```
 6558:             # Fallback: compute heading from vector velocity components (x_vel, y_vel)
 6559:             try:
 6560:                 vals_deg = np.degrees(np.arctan2(self.y_vel, self.x_vel))
 6561:             except Exception:
 6562:                 vals_deg = np.zeros(self.num_agents, dtype=float)
 6563: 
 6564:             # Convert degrees to radians and shift by 180 degrees (consistent with raster logic)
```

## Line 6629
```
 6626:             sample_ds = next(iter(self.hdf5['refugia'].values()))
 6627:             max_row = sample_ds.shape[0] - 1
 6628:             max_col = sample_ds.shape[1] - 1
 6629:         except Exception:
 6630:             max_row = int(np.round(self.height / self.refugia_cell_size))
 6631:             max_col = int(np.round(self.width / self.refugia_cell_size))
 6632: 
```

## Line 6648
```
 6645:             ci = cols[valid].astype(int)
 6646:             try:
 6647:                 self.refugia_accumulator[ai, ri, ci] = cv[valid]
 6648:             except Exception:
 6649:                 # Fallback to safe per-agent assignment if vectorized write fails
 6650:                 for i in ai:
 6651:                     r = int(rows[i])
```

## Line 6655
```
 6652:                     c = int(cols[i])
 6653:                     try:
 6654:                         self.refugia_accumulator[i, r, c] = float(cv[i])
 6655:                     except Exception:
 6656:                         continue
 6657: 
 6658:         # Flushing to HDF5 happens in `timestep_flush` in batch.
```

## Line 6729
```
 6726:             # Provide safe fallbacks for both arrays
 6727:             self.closest_agent = np.full(self.num_agents, np.nan)
 6728:             self.nearest_neighbor_distance = np.full(self.num_agents, np.nan)
 6729:         except Exception:
 6730:             logger.exception('Unexpected error during neighbor init; re-raising')
 6731:             raise
 6732: 
```

## Line 6783
```
 6780:             except (KeyError, IndexError, ValueError, OSError) as e:
 6781:                 try:
 6782:                     logger.exception('HECRAS mapping failed for node fields; disabling HECRAS mapping for this sim: %s', e)
 6783:                 except Exception as e:
 6784:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=6780)
 6785:                     pass
 6786:                 self.use_hecras = False
```

## Line 6787
```
 6784:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=6780)
 6785:                     pass
 6786:                 self.use_hecras = False
 6787:             except Exception:
 6788:                 try:
 6789:                     logger.exception('Unexpected error during HECRAS mapping — re-raising')
 6790:                 except Exception as e:
```

## Line 6790
```
 6787:             except Exception:
 6788:                 try:
 6789:                     logger.exception('Unexpected error during HECRAS mapping — re-raising')
 6790:                 except Exception as e:
 6791:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=6786)
 6792:                     pass
 6793:                 raise
```

## Line 6821
```
 6818:                 except (ValueError, TypeError, KeyError, IndexError, OSError) as e:
 6819:                     try:
 6820:                         logger.exception("precompute_pixel_indices: get_inv_transform or geo_to_pixel_from_inv failed for key=%s: %s", key, e)
 6821:                     except Exception as e:
 6822:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=6816)
 6823:                         pass
 6824:                     rows, cols = geo_to_pixel(X, Y, transform)
```

## Line 6825
```
 6822:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=6816)
 6823:                         pass
 6824:                     rows, cols = geo_to_pixel(X, Y, transform)
 6825:                 except Exception:
 6826:                     try:
 6827:                         logger.exception('precompute_pixel_indices: unexpected error — re-raising')
 6828:                     except Exception as e:
```

## Line 6828
```
 6825:                 except Exception:
 6826:                     try:
 6827:                         logger.exception('precompute_pixel_indices: unexpected error — re-raising')
 6828:                     except Exception as e:
 6829:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=6828)
 6830:                         pass
 6831:                     raise
```

## Line 6903
```
 6900:                     self.y_vel = np.zeros(self.num_agents)
 6901:                 self.x_vel += vel_jitter[:, 0]
 6902:                 self.y_vel += vel_jitter[:, 1]
 6903:             except Exception as e:
 6904:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=6902)
 6905:                 pass
 6906:         except Exception as e:
```

## Line 6906
```
 6903:             except Exception as e:
 6904:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=6902)
 6905:                 pass
 6906:         except Exception as e:
 6907:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=6904)
 6908:             pass
 6909:                 
```

## Line 6943
```
 6940:         except (ValueError, TypeError, IndexError, AttributeError) as e:
 6941:             logger.exception('KDTree build failed for agent positions; providing empty neighbor results')
 6942:             tree = None
 6943:         except Exception:
 6944:             logger.exception('Unexpected error building KDTree; re-raising')
 6945:             raise
 6946: 
```

## Line 7032
```
 7029:         if np.any(kcal < 0):
 7030:             try:
 7031:                 logger.warning("Negative kcal detected!")
 7032:             except Exception as e:
 7033:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=7029)
 7034:                 pass
 7035:     
```

## Line 7669
```
 7666:             except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
 7667:                 try:
 7668:                     logger.exception('Numba drag wrapper failed with runtime issue; falling back to Python compute_drags: %s', e)
 7669:                 except Exception as e:
 7670:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=7665)
 7671:                     pass
 7672:                 # fallback
```

## Line 7674
```
 7671:                     pass
 7672:                 # fallback
 7673:                 out = compute_drags(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, self.simulation.wave_drag, self.simulation.swim_behav)
 7674:             except Exception:
 7675:                 logger.exception('Unexpected error in drag wrapper; re-raising')
 7676:                 raise
 7677:             self.simulation.drag = out
```

## Line 7857
```
 7854:                 if np.any(np.isnan(error)):
 7855:                     try:
 7856:                         logger.warning('nan in error')
 7857:                     except Exception as e:
 7858:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=7852)
 7859:                         pass
 7860:                     sys.exit()
```

## Line 7907
```
 7904:             except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
 7905:                 try:
 7906:                     logger.exception('Numba swim_core helper failed (runtime); falling back to Python path: %s', e)
 7907:                 except Exception as e:
 7908:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=7907)
 7909:                     pass
 7910:                 dxdy = _swim_core_numba(fv0x, fv0y, accx, accy, pidx, pidy, tired_mask, dead_mask, mask, dt)
```

## Line 7911
```
 7908:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=7907)
 7909:                     pass
 7910:                 dxdy = _swim_core_numba(fv0x, fv0y, accx, accy, pidx, pidy, tired_mask, dead_mask, mask, dt)
 7911:             except Exception:
 7912:                 logger.exception('Unexpected error in swim_core numba helper; re-raising')
 7913:                 raise
 7914:                 
```

## Line 7971
```
 7968:                     logger.debug('JUMP DEBUG t=0: mask[:3]=%s, displacement[:3]=%s', mask[:3], displacement[:3])
 7969:                     logger.debug('JUMP DEBUG t=0: heading[:3]=%s, dx[:3]=%s, dy[:3]=%s', self.simulation.heading[:3], dx[:3], dy[:3])
 7970:                     logger.debug('JUMP DEBUG t=0: dxdy[:3]=%s', dxdy[:3])
 7971:                 except Exception as e:
 7972:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=7970)
 7973:                     pass
 7974:             
```

## Line 7978
```
 7975:             if np.any(dxdy > 3):
 7976:                 try:
 7977:                     logger.warning('Jump displacement unexpectedly large; check jump parameters')
 7978:                 except Exception as e:
 7979:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=7976)
 7980:                     pass
 7981:            
```

## Line 7993
```
 7990:             # convenience copies used throughout behavior methods
 7991:             try:
 7992:                 self.num_agents = int(getattr(simulation_object, 'num_agents', 0))
 7993:             except Exception:
 7994:                 self.num_agents = 0
 7995:             self.X = getattr(simulation_object, 'X', None)
 7996:             self.Y = getattr(simulation_object, 'Y', None)
```

## Line 8263
```
 8260:                                 from emergent.salmon_abm.sockeye import _safe_build_kdtree
 8261:                                 self.simulation._refuge_tree = _safe_build_kdtree(refuge_nodes, name='refuge_tree')
 8262:                                 self.simulation._refuge_nodes = refuge_nodes
 8263:                             except Exception:
 8264:                                 self.simulation._refuge_tree = None
 8265:                                 self.simulation._refuge_nodes = refuge_nodes
 8266:                         
```

## Line 8620
```
 8617:                                 thr_repr = float(threshold_m)
 8618:                             try:
 8619:                                 logger.debug("border_cue: %d agents too close, threshold=%s; sample forces (x,y): %s", n_close, thr_repr, list(zip(sample_fx, sample_fy)))
 8620:                             except Exception as e:
 8621:                                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=8617)
 8622:                                 pass
 8623:                         except Exception as e:
```

## Line 8623
```
 8620:                             except Exception as e:
 8621:                                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=8617)
 8622:                                 pass
 8623:                         except Exception as e:
 8624:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=8619)
 8625:                             pass
 8626: 
```

## Line 8897
```
 8894:                                                  k=3, ext=0)
 8895:                 body_depths = self.simulation.z / (self.simulation.body_depth / 100.)
 8896:                 self.simulation.wave_drag = np.where(body_depths >= 3, 1, wave_drag_fun(body_depths))
 8897:             except Exception:
 8898:                 # Fallback: if CSV not available, assume no additional wave drag
 8899:                 self.simulation.wave_drag = np.ones_like(self.simulation.z)
 8900:            
```

## Line 9192
```
 9189:                     # Combine heading alignment and SOG alignment weighted by sog_weight
 9190:                     alignment_array[:, 0] = (1.0 - sog_weight) * alignment_array[:, 0] + sog_weight * weight * v_hat_vel_x * no_school
 9191:                     alignment_array[:, 1] = (1.0 - sog_weight) * alignment_array[:, 1] + sog_weight * weight * v_hat_vel_y * no_school
 9192:                 except Exception:
 9193:                     # On any error, fall back to heading-only alignment
 9194:                     pass
 9195:             
```

## Line 9691
```
 9688:                     # store drags into simulation state for later use
 9689:                     try:
 9690:                         self.simulation.drag = drags_tmp
 9691:                     except Exception:
 9692:                         self.simulation.drag = np.ascontiguousarray(drags_tmp)
 9693:                 except Exception:
 9694:                     swim_speeds = np.linalg.norm(fish_velocities - water_velocities, axis=-1)
```

## Line 9693
```
 9690:                         self.simulation.drag = drags_tmp
 9691:                     except Exception:
 9692:                         self.simulation.drag = np.ascontiguousarray(drags_tmp)
 9693:                 except Exception:
 9694:                     swim_speeds = np.linalg.norm(fish_velocities - water_velocities, axis=-1)
 9695:             else:
 9696:                 swim_speeds = np.linalg.norm(fish_velocities - water_velocities, axis=-1)
```

## Line 9736
```
 9733:             except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
 9734:                 try:
 9735:                     logger.exception('Numba bout-distance helper failed (runtime); falling back to un-optimized call: %s', e)
 9736:                 except Exception as e:
 9737:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=9731)
 9738:                     pass
 9739:                 dist_travelled = _bout_distance_numba(self.simulation.prev_X, self.simulation.X, self.simulation.prev_Y, self.simulation.Y)
```

## Line 9740
```
 9737:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=9731)
 9738:                     pass
 9739:                 dist_travelled = _bout_distance_numba(self.simulation.prev_X, self.simulation.X, self.simulation.prev_Y, self.simulation.Y)
 9740:             except Exception:
 9741:                 logger.exception('Unexpected error in bout-distance numba helper; re-raising')
 9742:                 raise
 9743:             self.simulation.dist_per_bout += dist_travelled
```

## Line 9780
```
 9777:                 except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
 9778:                     try:
 9779:                         logger.exception('Numba time-to-fatigue helper failed with runtime issue; falling back: %s', e)
 9780:                     except Exception as e:
 9781:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=9780)
 9782:                         pass
 9783:                     ttf = _time_to_fatigue_numba(swim_speeds, mask_dict['prolonged'].astype(bool), mask_dict['sprint'].astype(bool), a_p, b_p, a_s, b_s)
```

## Line 9784
```
 9781:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=9780)
 9782:                         pass
 9783:                     ttf = _time_to_fatigue_numba(swim_speeds, mask_dict['prolonged'].astype(bool), mask_dict['sprint'].astype(bool), a_p, b_p, a_s, b_s)
 9784:                 except Exception:
 9785:                     logger.exception('Unexpected error in _time_to_fatigue_numba; re-raising')
 9786:                     raise
 9787:                 return ttf
```

## Line 9899
```
 9896:                 except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
 9897:                     try:
 9898:                         logger.exception('Numba merged battery kernel failed (runtime); falling back to pure-Python implementation: %s', e)
 9899:                     except Exception as e:
 9900:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=9898)
 9901:                         pass
 9902:                     # fallback to original numpy behavior
```

## Line 9910
```
 9907:                     safe = ttf0 != 0
 9908:                     battery[mask_non_sustained] *= np.where(safe, np.maximum(0.0, ttf1 / ttf0), 0.0)
 9909:                     self.simulation.battery = np.clip(battery, 0, 1)
 9910:                 except Exception:
 9911:                     logger.exception('Unexpected error in merged battery numba kernel; re-raising')
 9912:                     raise
 9913:             else:
```

## Line 9997
```
 9994:                     logger.info('battery: %s', np.round(self.battery,4))
 9995:                     logger.info('swim behavior: %s', self.swim_behav[0])
 9996:                     logger.info('swim mode: %s', self.swim_mode[0])
 9997:                 except Exception as e:
 9998:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=9995)
 9999:                     pass
10000: 
```

## Line 10004
```
10001:                 if np.any(self.simulation.swim_behav == 3):
10002:                     try:
10003:                         logger.warning('Error no longer counts, fatigued')
10004:                     except Exception as e:
10005:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10001)
10006:                         pass
10007:                     # Do not exit the host process; raise a RuntimeError to allow callers to handle
```

## Line 10045
```
10042:                     # set drag into simulation state using the preallocated buffer
10043:                     try:
10044:                         self.simulation.drag = drags_out
10045:                     except Exception:
10046:                         try:
10047:                             self.simulation.drag = np.ascontiguousarray(drags_out)
10048:                         except Exception:
```

## Line 10048
```
10045:                     except Exception:
10046:                         try:
10047:                             self.simulation.drag = np.ascontiguousarray(drags_out)
10048:                         except Exception:
10049:                             logger.exception('Failed to set simulation.drag from numba kernel output')
10050:                 except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
10051:                     try:
```

## Line 10053
```
10050:                 except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
10051:                     try:
10052:                         logger.exception('Numba swim/drag kernel failed with runtime issue; falling back to pure-Python path: %s', e)
10053:                     except Exception as e:
10054:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10049)
10055:                         pass
10056:                     swim_speeds = self.swim_speeds()
```

## Line 10062
```
10059:                     mask_dict['prolonged'] = np.where((self.simulation.max_s_U < bl_s) & (bl_s <= self.simulation.max_p_U), True, False)
10060:                     mask_dict['sprint'] = np.where(bl_s > self.simulation.max_p_U, True, False)
10061:                     mask_dict['sustained'] = bl_s <= self.simulation.max_s_U
10062:                 except Exception:
10063:                     logger.exception('Unexpected error in merged swim/drag numba kernel; re-raising')
10064:                     raise
10065:             # end compiled-core try: add runtime fallback for outer try
```

## Line 10069
```
10066:             except (ValueError, TypeError, IndexError, AttributeError, OSError) as e:
10067:                 try:
10068:                     logger.exception('Compiled swim/drag/fatigue path failed at runtime; falling back to pure-Python path: %s', e)
10069:                 except Exception as e:
10070:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10064)
10071:                     pass
10072:                 # fallback: compute swim speeds and masks using Python implementations
```

## Line 10113
```
10110:                 try:
10111:                     self.simulation.drag = new_drags
10112:                     self.simulation.battery = new_batt
10113:                 except Exception:
10114:                     self.simulation.drag = np.ascontiguousarray(new_drags)
10115:                     self.simulation.battery = np.ascontiguousarray(new_batt)
10116:             except (ValueError, TypeError, IndexError, AttributeError):
```

## Line 10120
```
10117:                 logger.exception("_drag_and_battery_numba failed; falling back to Python calc_battery")
10118:                 # fallback: previous behavior
10119:                 self.calc_battery(per_rec, ttf,  mask_dict)
10120:             except Exception:
10121:                 logger.exception('Unexpected error in _drag_and_battery_numba; re-raising')
10122:                 raise
10123:             
```

## Line 10156
```
10153:         if not getattr(self, 'use_hecras', False) or not getattr(self, 'hecras_mapping_enabled', False):
10154:             try:
10155:                 self.precompute_pixel_indices()
10156:             except Exception:
10157:                 # If precompute fails, do not block timestep (fallback to on-demand geo_to_pixel)
10158:                 try:
10159:                     logger.exception("precompute_pixel_indices failed during timestep; falling back to on-demand geo_to_pixel")
```

## Line 10160
```
10157:                 # If precompute fails, do not block timestep (fallback to on-demand geo_to_pixel)
10158:                 try:
10159:                     logger.exception("precompute_pixel_indices failed during timestep; falling back to on-demand geo_to_pixel")
10160:                 except Exception as e:
10161:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10160)
10162:                     pass
10163:         
```

## Line 10288
```
10285:                     # scale impulse by timestep to convert to velocity-like change
10286:                     self.x_vel += impulse[:, 0] * dt_safe
10287:                     self.y_vel += impulse[:, 1] * dt_safe
10288:         except Exception as e:
10289:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10287)
10290:             pass
10291:         
```

## Line 10338
```
10335:                         self.vel_mag = np.sqrt(self.x_vel**2 + self.y_vel**2)
10336:                     if 'distance_to' in self.hecras_node_fields:
10337:                         self.distance_to = self.apply_hecras_mapping(self.hecras_node_fields['distance_to'])
10338:             except Exception as e:
10339:                 # Log error - this shouldn't fail silently
10340:                 try:
10341:                     logger.exception('ERROR resampling HECRAS data: %s', e)
```

## Line 10342
```
10339:                 # Log error - this shouldn't fail silently
10340:                 try:
10341:                     logger.exception('ERROR resampling HECRAS data: %s', e)
10342:                 except Exception as e:
10343:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10340)
10344:                     pass
10345:                 # Revert to old positions on error
```

## Line 10353
```
10350:             # finalize timing and write a log line
10351:             try:
10352:                 stage_times['total_timestep'] = time.time() - t_start
10353:             except Exception:
10354:                 stage_times['total_timestep'] = 0.0
10355:             out_dir = os.path.join('outputs', 'rl_training')
10356:             try:
```

## Line 10358
```
10355:             out_dir = os.path.join('outputs', 'rl_training')
10356:             try:
10357:                 os.makedirs(out_dir, exist_ok=True)
10358:             except Exception as e:
10359:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10355)
10360:                 pass
10361:             log_path = os.path.join(out_dir, 'sim_profile.log')
```

## Line 10367
```
10364:                     f.write(f"t={t}, ")
10365:                     f.write(', '.join([f"{k}={v:.6f}" for k, v in stage_times.items()]))
10366:                     f.write('\n')
10367:             except Exception as e:
10368:                 _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10363)
10369:                 pass
10370:         
```

## Line 10434
```
10431:                                                            dataset.bounds[1],
10432:                                                            dataset.bounds[3]],
10433:                                                    cmap='viridis')
10434:                     except Exception:
10435:                         background = None
10436: 
10437:                     agent_pts, = ax.plot(self.X, self.Y, marker='o', ms=2, ls='', color='red')
```

## Line 10449
```
10446:                         # small pause to allow GUI event loop to update; cap pause to a reasonable minimum
10447:                         try:
10448:                             plt.pause(max(0.001, float(dt)))
10449:                         except Exception:
10450:                             time.sleep(max(0.001, float(dt)))
10451:                         try:
10452:                             logger.info('Time Step %s complete', i)
```

## Line 10453
```
10450:                             time.sleep(max(0.001, float(dt)))
10451:                         try:
10452:                             logger.info('Time Step %s complete', i)
10453:                         except Exception as e:
10454:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10448)
10455:                             pass
10456: 
```

## Line 10461
```
10458:                     try:
10459:                         self.hdf5.flush()
10460:                         self.hdf5.close()
10461:                     except Exception as e:
10462:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10461)
10463:                         pass
10464: 
```

## Line 10471
```
10468:                         self.timestep(i, dt, g, pid_controller)
10469:                         try:
10470:                             logger.info('Time Step %s complete', i)
10471:                         except Exception as e:
10472:                             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10470)
10473:                             pass
10474:         3. Initializes the plot for the simulation visualization.
```

## Line 10574
```
10571:                     self.timestep(i, dt, g, pid_controller)
10572:                     try:
10573:                         logger.info('Time Step %s complete', i)
10574:                     except Exception as e:
10575:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10572)
10576:                         pass
10577:                     
```

## Line 10592
```
10589:                 self.timestep(i, dt, g, pid_controller)
10590:                 try:
10591:                     logger.info('Time Step %s complete', i)
10592:                 except Exception as e:
10593:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10589)
10594:                     pass
10595: 
```

## Line 10599
```
10596:                 if i == range(n)[-1]:
10597:                     try:
10598:                         self.hdf5.close()
10599:                     except Exception as e:
10600:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10595)
10601:                         pass
10602:                     # Avoid forcing a hard exit; raise to allow callers to decide
```

## Line 10607
```
10604:             
10605:         try:
10606:             logger.info('ABM took %s to compile', (t1-t0))
10607:         except Exception as e:
10608:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10602)
10609:             pass
10610: 
```

## Line 10822
```
10819: 
10820:                                     bins = max(1, round((max(lengths_by_sex) - min(lengths_by_sex)) / bin_width))
10821:                                     ax.hist(lengths_by_sex, bins=bins, alpha=0.7, color='blue' if sex == 0 else 'pink')
10822:                                 except Exception as e:
10823:                                     try:
10824:                                         logger.exception('Error in calculating histogram for %s: %s', sex_label, e)
10825:                                     except Exception as e:
```

## Line 10825
```
10822:                                 except Exception as e:
10823:                                     try:
10824:                                         logger.exception('Error in calculating histogram for %s: %s', sex_label, e)
10825:                                     except Exception as e:
10826:                                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10825)
10827:                                         pass
10828:                                     continue
```

## Line 10839
```
10836:                             else:
10837:                                 try:
10838:                                     logger.warning('No length values found for %s.', sex_label)
10839:                                 except Exception as e:
10840:                                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10838)
10841:                                     pass
10842: 
```

## Line 10916
```
10913:                                     plt.tight_layout()
10914:                                     pdf.savefig(fig)
10915:                                     plt.close()
10916:                                 except Exception as e:
10917:                                     try:
10918:                                         logger.exception('Error in calculating histogram for %s: %s', sex_label, e)
10919:                                     except Exception as e:
```

## Line 10919
```
10916:                                 except Exception as e:
10917:                                     try:
10918:                                         logger.exception('Error in calculating histogram for %s: %s', sex_label, e)
10919:                                     except Exception as e:
10920:                                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10917)
10921:                                         pass
10922:                                     plt.close(fig)
```

## Line 10926
```
10923:                             else:
10924:                                 try:
10925:                                     logger.warning('No weight values found for %s in %s.', sex_label, base_name)
10926:                                 except Exception as e:
10927:                                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=10923)
10928:                                     pass
10929: 
```

## Line 11004
```
11001:                                     plt.tight_layout()
11002:                                     pdf.savefig(fig)
11003:                                     plt.close()
11004:                                 except Exception as e:
11005:                                     try:
11006:                                         logger.exception('Error in calculating histogram for %s: %s', sex_label, e)
11007:                                     except Exception as e:
```

## Line 11007
```
11004:                                 except Exception as e:
11005:                                     try:
11006:                                         logger.exception('Error in calculating histogram for %s: %s', sex_label, e)
11007:                                     except Exception as e:
11008:                                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=11003)
11009:                                         pass
11010:                                     plt.close(fig)
```

## Line 11014
```
11011:                             else:
11012:                                 try:
11013:                                     logger.warning('No body depth values found for %s in %s.', sex_label, base_name)
11014:                                 except Exception as e:
11015:                                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=11009)
11016:                                     pass
11017: 
```

## Line 11204
```
11201:                         all_male_kcals.extend(male_total_kcal)
11202:                         all_female_kcals.extend(female_total_kcal)
11203:     
11204:             except Exception as e:
11205:                 try:
11206:                     logger.exception('Failed to process %s: %s', hdf_path, e)
11207:                 except Exception as e:
```

## Line 11207
```
11204:             except Exception as e:
11205:                 try:
11206:                     logger.exception('Failed to process %s: %s', hdf_path, e)
11207:                 except Exception as e:
11208:                     _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=11207)
11209:                     pass
11210:     
```

## Line 11237
```
11234:     
11235:         try:
11236:             logger.info('Kcal histograms saved to %s and %s', male_histogram_path, female_histogram_path)
11237:         except Exception as e:
11238:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=11236)
11239:             pass
11240:             
```

## Line 11488
```
11485:                     data_over_time[timestep, :, :] = agent_counts_grid
11486:                     try:
11487:                         logger.info('file %s timestep %s complete', filename, timestep)
11488:                     except Exception as e:
11489:                         _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=11486)
11490:                         pass
11491:             
```

## Line 11523
```
11520:         
11521:         try:
11522:             logger.info('Dual band raster %s created successfully.', output_file)
11523:         except Exception as e:
11524:             _safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line=11520)
11525:             pass
11526:    
```

