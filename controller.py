import time
import json
import threading
from datetime import datetime as dt

from sources.Gemini import *
from data_processor import *
from models.cases import *
from models.users import *
from models.audit import add_audit_entry
from file_controller import *
from live_search.liveSearchController import *


def _audit_safe(
    *,
    user_id: str,
    case_id: str,
    kind: str,
    text: str = "",
    payload=None,
    status: str = "",
    source: str = "",
):
    """Best-effort audit write; never raises into product flows."""
    try:
        if not user_id or not case_id:
            return
        add_audit_entry(
            user_id=user_id,
            case_id=case_id,
            kind=kind,
            text=text,
            payload=payload or {},
            status=status,
            source=source,
        )
    except Exception as e:
        print(f"LOG: audit write skipped ({type(e).__name__}: {e})")


def _seed_search_limitations_before_create(case_data: dict) -> None:
    """
    After page isolation, before create_case: two-step seed via Gemini +
    add_limitations (classify then companies/URLs). Best-effort; never blocks
    patent import.
    """
    if not isinstance(case_data, dict):
        return
    try:
        from structures.limitations_seed import seed_search_limitations_on_fetch

        result = seed_search_limitations_on_fetch(case_data)
        lim = result.get("search_limitations") or {}
        print(
            "LOG: search_limitations seed result: "
            f"skipped={result.get('skipped')} "
            f"classify_attempts={result.get('classify_attempts')} "
            f"company_attempts={result.get('company_attempts')} "
            f"companies={len(result.get('accepted_companies') or [])} "
            f"urls={len(result.get('accepted_urls') or [])} "
            f"errors={len(result.get('errors') or [])} "
            f"layer={lim.get('commercial_layer')!r} "
            f"sellable_unit={lim.get('sellable_unit')!r}",
            flush=True,
        )
    except Exception as e:
        print(
            f"LOG: search_limitations seed failed ({type(e).__name__}: {e})",
            flush=True,
        )


def generate_patent_description(case_id):
    """
    Generate a description for a patent by case_id
    """
    case = get_case_by_id(case_id)
    if case is None:
        return {'success': False, 'message': 'Case not found'}
    
    document_urls = case.get('document_urls', [])
    document_contents = []
    for document in document_urls:
        content  = readDocumentFromUrl(document, headers={"X-API-KEY": getEnvKey('uspto')})
        document_contents.append(content)

    if (len(document_contents) == 0) or (document_contents is None):
        return {'success': False, 'message': 'No viable document contents provided'}
    
    complete_document_contents = ""
    for content in document_contents:
        if content is not None and str(content).strip() != "":
            complete_document_contents = f"{complete_document_contents}\n\n{content}"
    
    if complete_document_contents.strip() == "":
        return {'success': False, 'message': 'No viable document contents provided'}

    summary = get_patent_summary(complete_document_contents)
    
    # Update Case Data for the Generated Description
    result = update_case(case_id, {'description': summary, 'last_updated': dt.now()})
    return {
        'success': True, 
        'message': 'Patent summary generated successfully', 
        'summary': summary
        }


def _claims_already_populated(claims) -> bool:
    if claims is None:
        return False
    if isinstance(claims, dict):
        return len(claims) > 0
    if isinstance(claims, list):
        return len(claims) > 0
    return False


def isolate_claims(case_id):
    """
    Isolate the claims for a given case_id
    """
    case_data = get_case_by_id(case_id)
    if case_data is None:
      print(f'\nERROR: Error getting claims: Case not found')
      return {'success': False, 'message': 'Case not found'}

    existing_claims = case_data.get('claims', [])
    if _claims_already_populated(existing_claims):
      return {'success': True, 'message': 'Claims already exist', 'claims': existing_claims}

    description = case_data.get('description', '')
    complete_document_contents = ""
    if description.strip() != "":
      complete_document_contents = f"Description:\n{description}"
    
    document_urls = case_data.get('document_urls', [])
    document_contents = []
    for document in document_urls:
      if 'uspto' in document:
        content  = readDocumentFromUrl(document, headers={"X-API-KEY": getEnvKey('uspto')})
      elif '/document/' in document:
        doc_id = document.split('/')[-1].strip()
        content = readLocalDocument(doc_id)
      else:
        content = readDocumentFromUrl(document)
      document_contents.append(content)

    documents = case_data.get('documents', [])
    for document in documents:
      if document.get('source', '') == 'uspto':
        content  = readDocumentFromUrl(document.get('url', ''), headers={"X-API-KEY": getEnvKey('uspto')})
        document_contents.append(content)
      elif document.get('source', '') == 'local':
        doc_id = document.get('url', '').split('/')[-1].strip()
        document_view = getDocumentById(doc_id)
        if document_view.get('success', False):
          document_blob = document_view.get('document', {}).get('file_content', '')
          content = document_blob.decode('utf-8')
          document_contents.append(content)
      else:
        document_contents.append(document.get('content', ''))

    if (len(document_contents) == 0) or (document_contents is None):
      return {
        'success': False, 
        'message': 'No viable document contents provided', 
        'documents': {
          'document_urls_key': document_urls,
          'documents_key': document_contents
        }}

    complete_document_contents = ""
    for content in document_contents:
      if content.strip() != "":
        complete_document_contents = f"{complete_document_contents}\n\n{content}"
    
    if complete_document_contents.strip() == "":
      print(f'\nERROR: Error getting claims: No viable document contents provided')
      return {'success': False, 'message': 'No viable document contents provided'}

    try:
      isolated = Gemini().extract_claims(patent_content=complete_document_contents)
      validated, error_message = isolated.verify_isolated_claims()
      if not validated:
        print(f'\nERROR: Error getting claims: {error_message}')
        return {'success': False, 'message': error_message}
      claims = {
        str(i): isolated.claims[i].model_dump()
        for i in range(len(isolated.claims))
      }
    except Exception as e:
      print(f'\nERROR: Error extracting claims: {str(e)}')
      return {'success': False, 'message': str(e)}

    if not claims:
      print(f'\nERROR: Error getting claims: No claims found')
      return {'success': False, 'message': 'No claims found'}

    # Update Claims in Case Data
    result = update_case(case_id, {'claims': claims, 'last_updated': dt.now()})
    if result['success']:
      return {'success': True, 'message': 'Claims updated successfully', 'claims': claims}
    else:
      print(f'\nERROR: Error updating claims: {result["message"]}')
      return {'success': False, 'message': result['message']}


def fetchById(app, patent_id:str, user_id:str):
    patent_id = str(patent_id).strip().upper()
    uspto_error = False
    google_error = False
    free_patents_error = False
    error_message = ""
    fetch_succeeded = False
    isolation_started_at = dt.now().isoformat()
    with app.app_context():
        try:
            try:
                similar = get_similar_cases(patent_id, user_id)
                if similar and create_case(similar).get('success'):
                    remove_patent_from_fetching_list(user_id, patent_id)
                    remove_patent_from_error_list(user_id, patent_id)
                    print(f"LOG {patent_id}: Similar Cases Found and merged into a single case: {json.dumps(similar, indent=4)}")
                    case_id = similar['_id']
                    _audit_safe(
                        user_id=user_id,
                        case_id=case_id,
                        kind="patent_fetch",
                        text=f"Patent fetch completed for {patent_id} (similar-case merge)",
                        payload={
                            "patent_id": patent_id,
                            "started_at": isolation_started_at,
                            "source": "similar_cases",
                        },
                        status="completed",
                        source="fetch",
                    )
                    returnVal = {
                        'success': True,
                        'message': 'Patent data imported successfully',
                        'case_id': case_id,
                        'keywords': similar.get('keywords', []),
                        'case_data': similar,
                    }
                    fetch_succeeded = True
                    return returnVal, 200
                print(f"WARNING {patent_id}: No similar cases found, falling through to USPTO → Google → FPO")
            except Exception as e:
                print(f"ERROR: Error getting similar cases: {str(e)}")
            try:
                uspto_instance = USPTOPatentAPI(api_key=getEnvKey('uspto'))
                uspto_data = uspto_instance.get_complete_patent_info(patent_id)
                if uspto_data is None:
                    uspto_error = True
                    raise Exception("No Data found through USPTO")
                else:
                    uspto_data['created_by'] = user_id
                    uspto_data['_id'] = f"uspto_{user_id}_{patent_id}"
                    uspto_data['keywords'] = getKeywordsFromPatent(uspto_data['documents'])
                    print(f'\nUSPTO Data: {json.dumps(uspto_data, indent=4)}')
                    _seed_search_limitations_before_create(uspto_data)
                    creationResult = create_case(uspto_data)
                    print(f"TEST 1.1: Creation Result: {json.dumps(creationResult, indent=4)}")
                    if not creationResult.get('success'):
                        raise Exception(creationResult.get('message') or "Document Creation Error")
                    case_id = creationResult.get('case_id') or uspto_data['_id']
                    descriptionResult = generate_patent_description(case_id)
                    print(f"TEST 1.2: Description Generation Result: {json.dumps(descriptionResult, indent=4)}")
                    claimsResult = isolate_claims(case_id)
                    print(f"TEST 1.3: Claims Isolation Result: {json.dumps(claimsResult, indent=4)}")
                    remove_patent_from_fetching_list(user_id, patent_id)
                    remove_patent_from_error_list(user_id, patent_id)
                    _audit_safe(
                        user_id=user_id,
                        case_id=case_id,
                        kind="patent_fetch",
                        text=f"Patent fetch completed for {patent_id} (USPTO)",
                        payload={
                            "patent_id": patent_id,
                            "started_at": isolation_started_at,
                            "source": "uspto",
                        },
                        status="completed",
                        source="fetch",
                    )
                    fetch_succeeded = True
                    returnValue = {
                        'success': True,
                        'message': 'Patent data imported successfully',
                        'case_id': case_id,
                        'keywords': uspto_data.get('keywords', []),
                        'case_data': uspto_data
                    }
                    return returnValue, 200
            except Exception as e:
                print(f'\nError getting patent data from USPTO: {str(e)}')
                uspto_error = True
                error_message = str(e)
            # Try searching patent id using Google Patents
            if uspto_error:
                time.sleep(5)
                try:
                    google_patents = GooglePatents()
                    google_patents_details = google_patents.search_by_id(patent_id)
                    if google_patents_details is not None:
                        if isinstance(google_patents_details, dict):
                            extracted = passToGeminiForMetadata(
                                google_patents_details.get('metadata_content', ''),
                                claims_content=google_patents_details.get('claims_content'),
                                default_source='google_patents',
                            )
                        else:
                            extracted = passToGeminiForMetadata(
                                str(google_patents_details),
                                default_source='google_patents',
                            )
                        if extracted is None:
                            raise Exception(
                                "Failed to extract patent metadata from Google Patents content"
                            )
                        case_data = extracted.model_dump()
                        if case_data is not None:
                            patent_page_url = google_patents.get_patent_page_url(patent_id)
                            if patent_page_url:
                                case_data['document_urls'] = [patent_page_url]
                                case_data['documents'] = [
                                    {'url': patent_page_url, 'source': 'google_patents'}
                                ]
                            case_data['source'] = 'google_patents'
                            case_data['_id'] = f"googlepatents_{user_id}_{patent_id}"
                            case_data['created_by'] = user_id
                            if case_data.get('current_status', '') == '':
                                case_data['current_status'] = 'Granted'
                            case_data['created_date'] = dt.now().strftime('%Y-%m-%d')
                            created_id = case_data.get('_id', '')
                            _seed_search_limitations_before_create(case_data)
                            creationResult = create_case(case_data)
                            print(f"TEST 2.1: Creation Result: {json.dumps(creationResult, indent=4)}")
                            if not creationResult.get('success'):
                                raise Exception(creationResult.get('message') or "Document Creation Error")
                            created_id = creationResult.get('case_id') or case_data.get('_id', '')
                            if 'DocumentCreationError' in str(created_id):
                                raise Exception("Document Creation Error")
                            descriptionResult = generate_patent_description(created_id)
                            print(f"TEST 2.2: Description Generation Result: {json.dumps(descriptionResult, indent=4)}")
                            claimsResult = isolate_claims(created_id)
                            print(f"TEST 2.3: Claims Isolation Result: {json.dumps(claimsResult, indent=4)}")
                            remove_patent_from_fetching_list(user_id, patent_id)
                            remove_patent_from_error_list(user_id, patent_id)
                            _audit_safe(
                                user_id=user_id,
                                case_id=created_id,
                                kind="patent_fetch",
                                text=f"Patent fetch completed for {patent_id} (Google Patents)",
                                payload={
                                    "patent_id": patent_id,
                                    "started_at": isolation_started_at,
                                    "source": "google_patents",
                                },
                                status="completed",
                                source="fetch",
                            )
                            fetch_succeeded = True
                            returnValue = {
                                'success': True,
                                'message': 'Patent data imported successfully',
                                'case_id': created_id,
                                'keywords': case_data.get('keywords', []),
                                'case_data': case_data
                            }
                            return returnValue, 200
                    else:
                        raise Exception("No Data found through Google Patents")
                except Exception as e:
                    print(f"ERROR: Error getting patent details from Google Patents: {str(e)}")
                    error_message = str(e)
                    google_error = True
            # Patent not found using Google Patents, try using Free Patents Online
            if google_error and uspto_error:
                time.sleep(5)
                try:
                    free_patents = FreePatentsOnline()
                    free_patents_details = free_patents.search_by_id(patent_id)
                    if free_patents_details is not None:
                        extracted = passToGeminiForMetadata(
                            str(free_patents_details),
                            default_source='free_patents_online',
                        )
                        if extracted is None:
                            raise Exception(
                                "Failed to extract patent metadata from Free Patents Online content"
                            )
                        case_data = extracted.model_dump()
                        if case_data is not None:
                            case_data['source'] = 'free_patents_online'
                            case_data['_id'] = f"freepatentsonline_{user_id}_{patent_id}"
                            case_data['created_by'] = user_id
                            if case_data.get('current_status', '') == '':
                                case_data['current_status'] = 'Granted'
                            case_data['created_date'] = dt.now().strftime('%Y-%m-%d')
                            _seed_search_limitations_before_create(case_data)
                            creationResult = create_case(case_data)
                            print(f"TEST 3.1: Creation Result: {json.dumps(creationResult, indent=4)}")
                            if not creationResult.get('success'):
                                raise Exception(creationResult.get('message') or "Document Creation Error")
                            created_id = creationResult.get('case_id') or case_data.get('_id', '')
                            if 'DocumentCreationError' in str(created_id):
                                returnValue = {'success': False, 'message': created_id}
                                raise Exception("Document Creation Error")
                            descriptionResult = generate_patent_description(created_id)
                            print(f"TEST 3.2: Description Generation Result: {json.dumps(descriptionResult, indent=4)}")
                            claimsResult = isolate_claims(created_id)
                            print(f"TEST 3.3: Claims Isolation Result: {json.dumps(claimsResult, indent=4)}")
                            remove_patent_from_error_list(user_id, patent_id)
                            remove_patent_from_fetching_list(user_id, patent_id)
                            _audit_safe(
                                user_id=user_id,
                                case_id=created_id,
                                kind="patent_fetch",
                                text=f"Patent fetch completed for {patent_id} (Free Patents Online)",
                                payload={
                                    "patent_id": patent_id,
                                    "started_at": isolation_started_at,
                                    "source": "free_patents_online",
                                },
                                status="completed",
                                source="fetch",
                            )
                            fetch_succeeded = True
                            returnValue = {
                                'success': True,
                                'message': 'Patent data imported successfully',
                                'case_id': created_id,
                                'keywords': case_data.get('keywords', []),
                                'case_data': case_data
                            }
                            return returnValue, 200
                    else:
                        raise Exception("No Data found through Free Patents Online")
                except Exception as e:
                    print(f"ERROR: Error getting patent details from Free Patents Online: {str(e)}")
                    error_message = str(e)
                    free_patents_error = True

            errorReturn = {
                'success': False,
                'message': f"Failed to find patent with ID {patent_id}",
                'error_message': error_message
            }
            return errorReturn, 500
        finally:
            if not fetch_succeeded:
                try:
                    set_patent_to_error_list(
                        user_id,
                        patent_id,
                        error_message or 'Failed to import patent',
                    )
                except Exception as cleanup_err:
                    print(f'ERROR: Failed to update fetch error state: {cleanup_err}')
                    remove_patent_from_fetching_list(user_id, patent_id)


def check_thread_status(thread_name: str):
  for thread in threading.enumerate():
    if thread.name == thread_name:
      if thread.is_alive():
        return True, "Thread is running"
      else:
        return False, "Thread is not running"
  return False, "Thread was not found"